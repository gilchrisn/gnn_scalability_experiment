/**
 * mprw_exec.cpp — Standalone MPRW materialization executable.
 *
 * Interface is identical to graph_prep (Exact/KMV):
 *   - Loads graph from .dat files via HeterGraph (same code path)  
 *   - Starts internal chrono timer AFTER graph load (same as Exact/KMV t_start)
 *   - Emits "time:<seconds>" to stdout (parsed identically by engine.py)
 *   - Wrapped with /usr/bin/time -v by exp3_inference.py for peak RSS
 *
 * This ensures timing and memory comparisons against Exact/KMV are apples-to-apples.
 *
 * Usage:
 *   mprw_exec materialize <dataset_dir> <rule_file> <output_file> <w> <seed>
 *   mprw_exec profile     <dataset_dir> <rule_file> <output_file> <max_w> <seed>
 *
 * materialize: runs w walks per source node, outputs adjacency list (same format
 *              as run_materialization in ReadAnyBURLRules.cpp).
 *
 * profile:     depth-first walk, outputs per-node cumulative unique neighbor
 *              counts as a CSV (n_source rows x max_w columns). Used to find
 *              the plateau w without calibration binary-search.
 */

// Pull in HeterGraph, Pattern, Peers, parse_first_rule_from_file.
// Same include as HUB/main.cpp — zero modification to HUB/.
#include "../HUB/ReadAnyBURLRules.cpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// PCG32 — minimal single-file PRNG (Melissa O'Neill, pcg-random.org)
// No std::mt19937 bloat, no PyTorch RNG overhead.
// ---------------------------------------------------------------------------
struct PCG32 {
    uint64_t state;
    uint64_t inc;   // must be odd

    PCG32(uint64_t seed, uint64_t seq = 1ULL) noexcept {
        state = 0ULL;
        inc   = (seq << 1ULL) | 1ULL;
        next_u32();
        state += seed;
        next_u32();
    }

    inline uint32_t next_u32() noexcept {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xsh = (uint32_t)(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = (uint32_t)(old >> 59u);
        return (xsh >> rot) | (xsh << ((-rot) & 31u));
    }

    // Lemire's fast unbiased bounded integer in [0, range).
    // Avoids slow division (no % operator in the hot path).
    inline uint32_t bounded(uint32_t range) noexcept {
        uint64_t m = (uint64_t)next_u32() * (uint64_t)range;
        uint32_t l = (uint32_t)m;
        if (l < range) {
            // Rejection sampling to remove modulo bias — rare branch.
            uint32_t threshold = (uint32_t)(-(int32_t)range) % range;
            while (l < threshold) {
                m = (uint64_t)next_u32() * (uint64_t)range;
                l = (uint32_t)m;
            }
        }
        return (uint32_t)(m >> 32);
    }
};

// ---------------------------------------------------------------------------
// StepAccess — per-hop neighbor lookup using NGET/rNGET guards.
//
// NGET[u] is a list of Guard{etype, begin, end} where [begin, end) are
// absolute indices into ETL[etype] (the flattened neighbor list for that
// edge type). This mirrors the CSR structure used by the Python kernel.
// Raw pointers are extracted once before the walk loop — zero per-step
// overhead, no vector bounds checks in the inner loop.
// ---------------------------------------------------------------------------
struct StepAccess {
    const unsigned int* dst;        // raw ptr into ETL[etype] or rETL[etype]
    const uint32_t*     beg_ptr;    // beg_[u] = guard.begin for node u
    const uint32_t*     end_ptr;    // end_[u] = guard.end   for node u
    uint32_t            n_nodes;

    // Backing storage (kept alive while StepAccess is live)
    std::vector<uint32_t> beg_;
    std::vector<uint32_t> end_;

    inline uint32_t degree(uint32_t u) const noexcept {
        return end_ptr[u] - beg_ptr[u];
    }

    // Uniformly random neighbor of u — caller guarantees degree(u) > 0.
    inline uint32_t random_nbr(uint32_t u, PCG32& rng) const noexcept {
        return dst[beg_ptr[u] + rng.bounded(end_ptr[u] - beg_ptr[u])];
    }
};

static StepAccess build_step(const HeterGraph& g, int etype, int edirect) {
    StepAccess acc;
    acc.n_nodes = (uint32_t)g.NT.size();
    acc.beg_.assign(acc.n_nodes, 0);
    acc.end_.assign(acc.n_nodes, 0);

    // Forward (EDirect==1): use NGET + ETL.  Reverse: use rNGET + rETL.
    const auto& guards = (edirect == 1) ? g.NGET : g.rNGET;
    const auto& etl    = (edirect == 1) ? g.ETL  : g.rETL;

    if ((uint32_t)etype >= etl.size() || etl[etype] == nullptr) {
        acc.dst     = nullptr;
        acc.beg_ptr = acc.beg_.data();
        acc.end_ptr = acc.end_.data();
        return acc;   // all degrees 0 — walks die at this hop
    }
    acc.dst = etl[etype]->data();

    // Fill per-node (begin, end) from NGET guards.
    // Nodes without this edge type keep (0, 0) → degree = 0.
    for (uint32_t u = 0; u < acc.n_nodes; u++) {
        for (const auto& guard : *guards[u]) {
            if ((int)guard.etype == etype) {
                acc.beg_[u] = guard.begin;
                acc.end_[u] = guard.end;
                break;
            }
        }
    }
    acc.beg_ptr = acc.beg_.data();
    acc.end_ptr = acc.end_.data();
    return acc;
}

// ---------------------------------------------------------------------------
// Source node collection — mirrors Peers() variable-mode logic:
// collect all nodes whose NT contains NTypes[0] (the source type of the
// metapath).  NTypes[0] == -1 means any type.
// ---------------------------------------------------------------------------
static std::vector<uint32_t> collect_sources(const HeterGraph& g, const Pattern* qp) {
    int src_type = qp->NTypes.empty() ? -1 : qp->NTypes[0];
    std::vector<uint32_t> sources;
    sources.reserve(g.NT.size() / 4);
    for (uint32_t u = 0; u < (uint32_t)g.NT.size(); u++) {
        if (src_type == -1) {
            sources.push_back(u);
            continue;
        }
        for (unsigned int t : *g.NT[u]) {
            if ((int)t == src_type) { sources.push_back(u); break; }
        }
    }
    return sources;
}

// ---------------------------------------------------------------------------
// Write adjacency list — same format as run_materialization output:
//   "<src> <nbr1> <nbr2> ...\n"   (space-separated, sorted neighbors)
// Only nodes with at least one neighbor are written.
// ---------------------------------------------------------------------------
static void write_adj(const std::string& path,
                      const std::vector<std::vector<uint32_t>>& adj,
                      uint32_t n_nodes) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "[mprw_exec] ERROR: cannot open output file: " << path << "\n";
        std::exit(1);
    }
    for (uint32_t u = 0; u < n_nodes; u++) {
        if (adj[u].empty()) continue;
        out << u;
        for (uint32_t v : adj[u]) out << ' ' << v;
        out << '\n';
    }
}

// ---------------------------------------------------------------------------
// PATH 1 — materialize
//
// Loop order: BFS (w-epoch outer, node inner) — uniform degradation if memory
// capped, plus dedup after every epoch keeps peak working-set bounded.
//
// Memory accounting tracks only algorithm-allocated edge arrays, same
// principle as Gemini's spec but using vector::capacity() as the source of
// truth (matches what the OS will actually page in).
// ---------------------------------------------------------------------------
static void cmd_materialize(const std::string& dataset,
                            const std::string& rule_file,
                            const std::string& output_file,
                            int64_t max_w,
                            uint64_t seed,
                            int64_t max_memory_bytes) {
    // ── Graph + rule load (NOT timed — same exclusion as Exact/KMV) ──────────
    HeterGraph g(dataset);
    Pattern* qp = parse_first_rule_from_file(rule_file);

    uint32_t n_nodes = (uint32_t)g.NT.size();
    int      n_hops  = (int)qp->ETypes.size();

    // Build per-hop StepAccess arrays (once, reused across all walks)
    std::vector<StepAccess> steps(n_hops);
    for (int h = 0; h < n_hops; h++)
        steps[h] = build_step(g, qp->ETypes[h], qp->EDirect[h]);

    std::vector<uint32_t> sources = collect_sources(g, qp);
    uint32_t n_src = (uint32_t)sources.size();

    // ── Algorithmic timer starts here — identical to t_start in HUB code ────
    auto t_start = std::chrono::steady_clock::now();

    // Global edge set — undirected, self-loops excluded.
    // Using sorted vectors per node for O(log n) dedup; unordered_set would be
    // faster to insert but has worse cache behavior during the walk loop.
    std::vector<std::vector<uint32_t>> adj(n_nodes);

    // Temporary per-epoch edge buffer.  Reused across epochs to avoid
    // re-allocation; cleared at the start of each epoch.
    // Size: one terminal per (source, walk) pair in this epoch.
    std::vector<uint32_t> ep_src;
    std::vector<uint32_t> ep_dst;
    ep_src.reserve(n_src);
    ep_dst.reserve(n_src);

    for (int64_t w = 0; w < max_w; w++) {
        ep_src.clear();
        ep_dst.clear();

        // ── Inner loop: all source nodes for walk w ─────────────────────────
        // Each source gets its own PCG32 seeded by (seed, src_idx * max_w + w)
        // so walk w of source u is fully independent of all other walks.
        for (uint32_t si = 0; si < n_src; si++) {
            uint32_t u    = sources[si];
            uint32_t cur  = u;

            // Per-walk RNG: unique stream per (source, walk) tuple.
            // Uses a large prime (1000003) instead of max_w so that
            // edges(w=8) ⊇ edges(w=4) — prefix-consistent across w values.
            PCG32 rng(seed, (uint64_t)si * 1000003ULL + (uint64_t)w + 1);

            bool alive = true;
            for (int h = 0; h < n_hops && alive; h++) {
                uint32_t deg = steps[h].degree(cur);
                if (deg == 0) { alive = false; break; }
                cur = steps[h].random_nbr(cur, rng);
            }

            if (alive && cur != u) {   // valid terminal, not a self-loop
                ep_src.push_back(u);
                ep_dst.push_back(cur);
            }
        }

        // ── Epoch compaction: merge epoch edges into adj, dedup immediately ─
        // Sorting and deduplicating at every epoch (not just at the end)
        // prevents unbounded RAM growth — each adj[u] stays at true unique
        // neighbor count, not proportional to w * |sources|.
        for (size_t i = 0; i < ep_src.size(); i++) {
            uint32_t u = ep_src[i], v = ep_dst[i];
            // Directed edge u→v; we'll make undirected at the end.
            auto& nbrs = adj[u];
            // Insert in sorted position for efficient dedup.
            auto pos = std::lower_bound(nbrs.begin(), nbrs.end(), v);
            if (pos == nbrs.end() || *pos != v)
                nbrs.insert(pos, v);
        }

        // ── Memory check: break if projected peak would exceed limit ─────────
        if (max_memory_bytes > 0) {
            // Current allocation: all adj vectors' capacity * 4 bytes each.
            // (uint32_t = 4 bytes; two arrays for undirected coalesce below)
            int64_t adj_bytes = 0;
            for (uint32_t u = 0; u < n_nodes; u++)
                adj_bytes += (int64_t)adj[u].capacity() * sizeof(uint32_t);

            // Coalesce penalty: making undirected + sorting requires ~5x
            // transient arrays at peak.  Conservative estimate.
            int64_t total_edges = 0;
            for (uint32_t u = 0; u < n_nodes; u++)
                total_edges += (int64_t)adj[u].size();
            int64_t coalesce_penalty = (total_edges * 2 + n_nodes) * 5 * (int64_t)sizeof(uint32_t);

            if (adj_bytes + coalesce_penalty >= max_memory_bytes) {
                std::cerr << "[mprw_exec] memory cap hit at w=" << w
                          << " (" << (adj_bytes + coalesce_penalty) / (1024 * 1024)
                          << " MB projected, limit="
                          << max_memory_bytes / (1024 * 1024) << " MB)\n";
                break;
            }
        }
    }

    // ── Make undirected: for every (u,v) add (v,u) ──────────────────────────
    for (uint32_t u = 0; u < n_nodes; u++) {
        for (uint32_t v : adj[u]) {
            if (v == u) continue;   // skip self-loops
            auto& nbrs_v = adj[v];
            auto pos = std::lower_bound(nbrs_v.begin(), nbrs_v.end(), u);
            if (pos == nbrs_v.end() || *pos != u)
                nbrs_v.insert(pos, u);
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double algo_seconds = std::chrono::duration<double>(t_end - t_start).count();

    // ── Output ───────────────────────────────────────────────────────────────
    write_adj(output_file, adj, n_nodes);

    // Emit timing — identical protocol to HUB's "time:" line (parsed by engine.py).
    std::cout << "time:" << algo_seconds << std::endl;

    int64_t total_edges = 0;
    for (uint32_t u = 0; u < n_nodes; u++) total_edges += (int64_t)adj[u].size();
    std::cerr << "[mprw_exec] materialize done: sources=" << n_src
              << " w=" << max_w
              << " edges=" << total_edges
              << " time=" << algo_seconds << "s\n";

    delete qp;
}

// ---------------------------------------------------------------------------
// PATH 2 — profile (density-matched materialization)
//
// For each source node u, walk until its unique-neighbor count equals
// target_degree[u] (read from a KMV adjacency file), OR until the global
// peak memory of all adjacency vectors exceeds max_memory_bytes.
//
// No per-node walk cap — walks as long as needed. The memory limit is the
// only global safety valve. Use /usr/bin/time -v externally to measure
// true peak RSS.
//
// Answers the question: "how many random walks does MPRW need to match
// KMV density at k=X?"
//
// Stdout (one line each, in order):
//   time:<seconds>          — algo time (after graph load)
//   mean_w:<float>          — mean walks needed across saturated nodes
//   max_w_used:<int>        — max walks any single node needed
//   total_walks:<int>       — grand total walks executed
//
// Stderr: saturated/mem_stopped counts.
//
// Usage:
//   mprw_exec profile <dataset> <rule_file> <kmv_adj> <output_adj> <seed> [max_mem_mb]
// ---------------------------------------------------------------------------
static void cmd_profile(const std::string& dataset,
                        const std::string& rule_file,
                        const std::string& kmv_adj_file,
                        const std::string& output_file,
                        uint64_t seed,
                        int64_t max_memory_bytes) {
    // ── Read per-node KMV target degrees ─────────────────────────────────────
    std::unordered_map<uint32_t, uint32_t> target_degree;
    {
        std::ifstream kmv_f(kmv_adj_file);
        if (!kmv_f.is_open()) {
            std::cerr << "[mprw_exec] ERROR: cannot open KMV adj file: " << kmv_adj_file << "\n";
            std::exit(1);
        }
        std::string line;
        while (std::getline(kmv_f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            uint32_t node; ss >> node;
            uint32_t cnt = 0;
            uint32_t nbr;
            while (ss >> nbr) cnt++;
            if (cnt > 0) target_degree[node] = cnt;
        }
    }

    // ── Graph + rule load (NOT timed) ────────────────────────────────────────
    HeterGraph g(dataset);
    Pattern* qp = parse_first_rule_from_file(rule_file);

    uint32_t n_nodes = (uint32_t)g.NT.size();
    int      n_hops  = (int)qp->ETypes.size();

    std::vector<StepAccess> steps(n_hops);
    for (int h = 0; h < n_hops; h++)
        steps[h] = build_step(g, qp->ETypes[h], qp->EDirect[h]);

    std::vector<uint32_t> sources = collect_sources(g, qp);
    uint32_t n_src = (uint32_t)sources.size();

    // ── Algorithmic timer starts here ────────────────────────────────────────
    auto t_start = std::chrono::steady_clock::now();

    std::vector<std::vector<uint32_t>> adj(n_nodes);

    int64_t  total_saturated  = 0;
    int64_t  total_capped_mem = 0;  // nodes stopped because global mem limit hit
    int64_t  total_walks      = 0;
    int64_t  max_w_used       = 0;
    double   sum_w_saturated  = 0.0;  // for mean_w across saturated nodes

    bool mem_limit_hit = false;

    for (uint32_t si = 0; si < n_src && !mem_limit_hit; si++) {
        uint32_t u = sources[si];

        auto it = target_degree.find(u);
        if (it == target_degree.end()) continue;  // KMV gave this node no neighbors
        uint32_t target = it->second;

        std::unordered_set<uint32_t> seen;
        seen.reserve(target * 2);
        uint32_t found = 0;
        int64_t  w     = 0;

        // Walk until density matched — no per-node cap.
        // The only termination is: found >= target (success) or mem limit (global).
        while (found < target) {
            PCG32 rng(seed, (uint64_t)si * 1000003ULL + (uint64_t)w + 1);

            uint32_t cur = u;
            bool alive = true;
            for (int h = 0; h < n_hops && alive; h++) {
                uint32_t deg = steps[h].degree(cur);
                if (deg == 0) { alive = false; break; }
                cur = steps[h].random_nbr(cur, rng);
            }

            if (alive && cur != u && seen.find(cur) == seen.end()) {
                seen.insert(cur);
                found++;
            }
            w++;
            total_walks++;

            // Memory check every 1000 walks to avoid overhead in the hot path.
            if (max_memory_bytes > 0 && (w % 1000 == 0)) {
                int64_t adj_bytes = 0;
                for (uint32_t v = 0; v < n_nodes; v++)
                    adj_bytes += (int64_t)adj[v].capacity() * sizeof(uint32_t);
                // Add current node's seen set (not yet in adj) — bucket_count * ptr size
                adj_bytes += (int64_t)seen.bucket_count() * sizeof(void*)
                           + (int64_t)seen.size() * sizeof(uint32_t);
                if (adj_bytes >= max_memory_bytes) {
                    std::cerr << "[mprw_exec] memory cap hit at node " << u
                              << " w=" << w
                              << " (" << adj_bytes / (1024 * 1024) << " MB)\n";
                    mem_limit_hit = true;
                    break;
                }
            }
        }

        // Commit to adj (sorted)
        auto& nbrs = adj[u];
        nbrs.assign(seen.begin(), seen.end());
        std::sort(nbrs.begin(), nbrs.end());

        if (found >= target) {
            total_saturated++;
            sum_w_saturated += (double)w;
            if (w > max_w_used) max_w_used = w;
        } else {
            total_capped_mem++;
        }
    }

    // NOTE: No "make undirected" pass — same reasoning as cmd_match.
    // target_degree[u] is from the symmetric KMV adj: sum(targets) == KMV edge count.

    auto t_end = std::chrono::steady_clock::now();
    double algo_seconds = std::chrono::duration<double>(t_end - t_start).count();

    write_adj(output_file, adj, n_nodes);

    double mean_w = (total_saturated > 0) ? sum_w_saturated / total_saturated : 0.0;

    std::cout << "time:" << algo_seconds << "\n"
              << "mean_w:" << mean_w << "\n"
              << "max_w_used:" << max_w_used << "\n"
              << "total_walks:" << total_walks << std::endl;

    int64_t total_edges = 0;
    for (uint32_t u = 0; u < n_nodes; u++) total_edges += (int64_t)adj[u].size();
    std::cerr << "[mprw_exec] profile done: sources=" << n_src
              << " saturated=" << total_saturated
              << " mem_stopped=" << total_capped_mem
              << " edges=" << total_edges
              << " total_walks=" << total_walks
              << " mean_w=" << mean_w
              << " max_w=" << max_w_used
              << " time=" << algo_seconds << "s\n";

    delete qp;
}


// ---------------------------------------------------------------------------
// KMV sketch merge — O(|dst| + |src|) sorted merge, keep k smallest unique.
// Both dst and src must be sorted ascending. Result is sorted, deduped, ≤k.
// ---------------------------------------------------------------------------
static void merge_kmv(std::vector<uint32_t>& dst,
                      const std::vector<uint32_t>& src,
                      int k) {
    if (src.empty()) return;
    if (dst.empty()) {
        dst = src;
        if ((int)dst.size() > k) dst.resize(k);
        return;
    }
    std::vector<uint32_t> merged;
    merged.reserve(std::min((size_t)k, dst.size() + src.size()));

    size_t i = 0, j = 0;
    while (i < dst.size() && j < src.size() && (int)merged.size() < k) {
        if (dst[i] < src[j]) {
            if (merged.empty() || merged.back() != dst[i])
                merged.push_back(dst[i]);
            i++;
        } else if (dst[i] > src[j]) {
            if (merged.empty() || merged.back() != src[j])
                merged.push_back(src[j]);
            j++;
        } else {                          // equal — take once
            if (merged.empty() || merged.back() != dst[i])
                merged.push_back(dst[i]);
            i++; j++;
        }
    }
    while (i < dst.size() && (int)merged.size() < k) {
        if (merged.empty() || merged.back() != dst[i])
            merged.push_back(dst[i]);
        i++;
    }
    while (j < src.size() && (int)merged.size() < k) {
        if (merged.empty() || merged.back() != src[j])
            merged.push_back(src[j]);
        j++;
    }
    dst = std::move(merged);
}

// ---------------------------------------------------------------------------
// PATH 3 — KGRW  (KMV-Guided Random Walk)
//
// Hybrid materialization that uses KMV sketch propagation for the first
// floor(L/2) meta-path hops and short random walks for the remaining
// ceil(L/2) hops.  Combines KMV's uniform midpoint coverage with MPRW's
// cheap walk-based endpoint discovery.
//
// Phase 1: KMV sketch propagation (hops 0 .. mid_hop-1)
//   Hash each source, propagate k-min sketches level-by-level.
//   Midpoint nodes accumulate sorted sketches of ≤k source hashes.
//
// Phase 2: Short random walks (hops mid_hop .. n_hops-1)
//   For every midpoint with a non-empty sketch, do w' independent walks.
//
// Phase 3: Cross-product edge construction
//   For midpoint m: ∀ source s ∈ sketch[m], ∀ endpoint u ∈ walks[m]:
//     Ã[s,u] = 1.  Dedup + make undirected.
//
// Output format is identical to cmd_materialize (adjacency list).
// ---------------------------------------------------------------------------
static void cmd_kgrw(const std::string& dataset,
                     const std::string& rule_file,
                     const std::string& output_file,
                     int k_budget,
                     int w_prime,
                     uint64_t seed,
                     int64_t /*max_memory_bytes*/) {
    // ── Graph + rule load (NOT timed — same exclusion as Exact/KMV) ────────
    HeterGraph g(dataset);
    Pattern* qp = parse_first_rule_from_file(rule_file);

    uint32_t n_nodes = (uint32_t)g.NT.size();
    int      n_hops  = (int)qp->ETypes.size();
    int      mid_hop = n_hops / 2;             // KMV covers hops [0, mid_hop)
    int      walk_hops = n_hops - mid_hop;     // walks cover hops [mid_hop, n_hops)

    std::vector<StepAccess> steps(n_hops);
    for (int h = 0; h < n_hops; h++)
        steps[h] = build_step(g, qp->ETypes[h], qp->EDirect[h]);

    std::vector<uint32_t> sources = collect_sources(g, qp);
    uint32_t n_src = (uint32_t)sources.size();

    std::cerr << "[kgrw] n_nodes=" << n_nodes
              << " n_src=" << n_src
              << " n_hops=" << n_hops
              << " mid_hop=" << mid_hop
              << " walk_hops=" << walk_hops << "\n";

    // ── Algorithmic timer starts here ──────────────────────────────────────
    auto t_start = std::chrono::steady_clock::now();

    // ── Phase 1: KMV Sketch Propagation ────────────────────────────────────
    // Hash each source node; store reverse mapping hash → node ID.
    std::unordered_map<uint32_t, uint32_t> hash_to_node;
    hash_to_node.reserve(n_src * 2);

    // Dense sketch storage: sketch[u] = sorted vector of ≤k hash values.
    std::vector<std::vector<uint32_t>> cur_sketch(n_nodes);

    PCG32 hash_rng(seed, 0);
    for (uint32_t si = 0; si < n_src; si++) {
        uint32_t u = sources[si];
        uint32_t h;
        do { h = hash_rng.next_u32(); } while (hash_to_node.count(h));
        hash_to_node[h] = u;
        cur_sketch[u].push_back(h);           // single-element sketch
    }

    auto t_hash = std::chrono::steady_clock::now();
    std::cerr << "[kgrw] phase1 hash: "
              << std::chrono::duration<double>(t_hash - t_start).count() << "s\n";

    // Level-by-level propagation through hops 0 .. mid_hop-1.
    for (int hop = 0; hop < mid_hop; hop++) {
        std::vector<std::vector<uint32_t>> next_sketch(n_nodes);

        for (uint32_t u = 0; u < n_nodes; u++) {
            if (cur_sketch[u].empty()) continue;
            uint32_t deg = steps[hop].degree(u);
            for (uint32_t i = 0; i < deg; i++) {
                uint32_t v = steps[hop].dst[steps[hop].beg_ptr[u] + i];
                merge_kmv(next_sketch[v], cur_sketch[u], k_budget);
            }
        }

        uint32_t n_active = 0;
        for (uint32_t u = 0; u < n_nodes; u++)
            if (!next_sketch[u].empty()) n_active++;

        std::cerr << "[kgrw] phase1 hop " << hop
                  << " → " << n_active << " active nodes at next level\n";
        cur_sketch = std::move(next_sketch);
    }

    auto t_prop = std::chrono::steady_clock::now();
    std::cerr << "[kgrw] phase1 propagation: "
              << std::chrono::duration<double>(t_prop - t_hash).count() << "s\n";

    // ── Phase 2+3: Walk & cross-product pair.
    //
    //    For midpoint m with sketch S_m (|S_m| ≤ k sources) and w' walks:
    //      Each walk j reaches endpoint u.
    //      Emit edge (s, u) for EVERY s in S_m — cross-product, not round-robin.
    //
    //    Theoretical basis: coupon-collector universe shrinks from d_S*d_T
    //    (MPRW) to d_T (KGRW), saving a factor of d_S walks.  Each walk is
    //    multiplied across all k sources in the sketch at zero extra walk cost.
    // ────────────────────────────────────────────────────────────────────────
    std::vector<std::vector<uint32_t>> adj(n_nodes);

    uint32_t n_mid_active = 0;
    uint32_t total_walks  = 0;
    uint32_t dead_walks   = 0;

    for (uint32_t m = 0; m < n_nodes; m++) {
        if (cur_sketch[m].empty()) continue;
        n_mid_active++;

        for (int w = 0; w < w_prime; w++) {
            PCG32 rng(seed + 1, (uint64_t)m * 1000003ULL + (uint64_t)w + 1);

            uint32_t cur = m;
            bool alive = true;
            for (int h = mid_hop; h < n_hops && alive; h++) {
                uint32_t deg = steps[h].degree(cur);
                if (deg == 0) { alive = false; break; }
                cur = steps[h].random_nbr(cur, rng);
            }
            total_walks++;

            if (!alive) { dead_walks++; continue; }

            uint32_t u = cur;
            // Cross-product: pair endpoint u with EVERY source in sketch.
            for (uint32_t hash_val : cur_sketch[m]) {
                auto src_it = hash_to_node.find(hash_val);
                if (src_it == hash_to_node.end()) continue;
                uint32_t s = src_it->second;
                if (u == s) continue;  // no self-loops
                auto& nbrs = adj[s];
                auto pos = std::lower_bound(nbrs.begin(), nbrs.end(), u);
                if (pos == nbrs.end() || *pos != u)
                    nbrs.insert(pos, u);
            }
        }
    }

    auto t_build = std::chrono::steady_clock::now();
    std::cerr << "[kgrw] phase2+3 walk&pair: "
              << std::chrono::duration<double>(t_build - t_prop).count() << "s"
              << "  midpoints=" << n_mid_active
              << "  total_walks=" << total_walks
              << "  dead=" << dead_walks << "\n";

    // ── Make undirected ─────────────────────────────────────────────────────
    for (uint32_t u = 0; u < n_nodes; u++) {
        for (uint32_t v : adj[u]) {
            if (v == u) continue;
            auto& nbrs_v = adj[v];
            auto pos = std::lower_bound(nbrs_v.begin(), nbrs_v.end(), u);
            if (pos == nbrs_v.end() || *pos != u)
                nbrs_v.insert(pos, u);
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double algo_seconds = std::chrono::duration<double>(t_end - t_start).count();

    // ── Output ─────────────────────────────────────────────────────────────
    write_adj(output_file, adj, n_nodes);
    std::cout << "time:" << algo_seconds << std::endl;

    int64_t total_edges = 0;
    for (uint32_t u = 0; u < n_nodes; u++) total_edges += (int64_t)adj[u].size();

    std::cerr << "[kgrw] DONE  sources=" << n_src
              << "  midpoints=" << n_mid_active
              << "  k=" << k_budget << "  w'=" << w_prime
              << "  edges=" << total_edges
              << "  time=" << algo_seconds << "s\n";

    delete qp;
}


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  mprw_exec materialize <dataset> <rule_file> <output> <w> <seed> [max_mem_mb]\n"
                  << "  mprw_exec profile     <dataset> <rule_file> <kmv_adj> <output> <seed> [max_mem_mb]\n"
                  << "  mprw_exec kgrw        <dataset> <rule_file> <output> <k> <w_prime> <seed> [max_mem_mb]\n";
        return 1;
    }

    std::string mode      = argv[1];
    std::string dataset   = argv[2];
    std::string rule_file = argv[3];

    if (mode == "materialize") {
        if (argc < 7) {
            std::cerr << "Usage: mprw_exec materialize <dataset> <rule_file> <output> <w> <seed> [max_mem_mb]\n";
            return 1;
        }
        std::string output  = argv[4];
        int64_t  w_or_maxw  = std::stoll(argv[5]);
        uint64_t seed       = (uint64_t)std::stoll(argv[6]);
        int64_t  max_mem_mb = (argc >= 8) ? std::stoll(argv[7]) : 0;
        cmd_materialize(dataset, rule_file, output, w_or_maxw, seed,
                        max_mem_mb > 0 ? max_mem_mb * 1024LL * 1024LL : 0LL);
    } else if (mode == "profile") {
        if (argc < 7) {
            std::cerr << "Usage: mprw_exec profile <dataset> <rule_file> <kmv_adj> <output> <seed> [max_mem_mb]\n";
            return 1;
        }
        std::string kmv_adj = argv[4];
        std::string output  = argv[5];
        uint64_t seed       = (uint64_t)std::stoll(argv[6]);
        int64_t  max_mem_mb = (argc >= 8) ? std::stoll(argv[7]) : 0;
        cmd_profile(dataset, rule_file, kmv_adj, output, seed,
                    max_mem_mb > 0 ? max_mem_mb * 1024LL * 1024LL : 0LL);
    } else if (mode == "kgrw") {
        if (argc < 8) {
            std::cerr << "Usage: mprw_exec kgrw <dataset> <rule_file> <output> <k> <w_prime> <seed> [max_mem_mb]\n";
            return 1;
        }
        std::string output  = argv[4];
        int k_budget        = std::stoi(argv[5]);
        int w_prime         = std::stoi(argv[6]);
        uint64_t seed       = (uint64_t)std::stoll(argv[7]);
        int64_t  max_mem_mb = (argc >= 9) ? std::stoll(argv[8]) : 0;
        cmd_kgrw(dataset, rule_file, output, k_budget, w_prime, seed,
                 max_mem_mb > 0 ? max_mem_mb * 1024LL * 1024LL : 0LL);
    } else {
        std::cerr << "[mprw_exec] Unknown mode: " << mode << "\n";
        return 1;
    }
    return 0;
}
