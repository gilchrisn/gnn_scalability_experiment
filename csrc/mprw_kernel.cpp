/**
 * mprw_kernel.cpp — Cache-optimised Metapath Random Walk materialisation.
 *
 * Design constraints (from the research spec):
 *   1. Absolute single-threading — no OpenMP, no std::thread, no ATen parallel.
 *   2. De-vectorised execution — outer loop over target nodes, inner loop over
 *      k walks, innermost loop over metapath hops.  All walker state lives in
 *      registers / L1 cache.
 *   3. Lightweight PRNG — PCG32 (minimal variant), not std::mt19937.
 *      Neighbour selection via Lemire's fast unbiased range reduction (no modulo).
 *   4. Zero inner-loop allocation — terminal_nodes buffer pre-allocated once and
 *      reused for every target node.
 *   5. Output via pre-allocated std::vector<int64_t> + memcpy into torch::Tensor.
 *
 * Build:
 *   python setup_mprw.py build_ext --inplace
 */

#include <torch/extension.h>

#include <algorithm>   // std::sort
#include <cstdint>
#include <cstring>     // std::memcpy
#include <vector>

// ---------------------------------------------------------------------------
// Disable ATen parallelism at compile time.
// ---------------------------------------------------------------------------
#ifndef AT_PARALLEL_OPENMP
#define AT_PARALLEL_OPENMP 0
#endif

// ---------------------------------------------------------------------------
// PCG32 — minimal permuted-congruential generator (O'Neill 2014).
//
// State: 16 bytes (two uint64_t).  Period: 2^64.
// Quality: passes BigCrush / PractRand.  Far lighter than mt19937 (2.5 kB).
// ---------------------------------------------------------------------------
struct PCG32 {
    uint64_t state;
    uint64_t inc;  // stream selector — must be odd

    // Seed the generator.  The stream is derived from the seed so that
    // different seeds yield independent sequences.
    inline void seed(uint64_t s) {
        state = 0u;
        inc   = (s << 1u) | 1u;  // guarantee odd
        next();                   // advance past the zero state
        state += s;
        next();                   // mix
    }

    // Generate a uniformly distributed uint32_t.
    inline uint32_t next() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xorshifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = static_cast<uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
    }

    // Lemire's nearly-divisionless unbiased range reduction.
    // Returns a uniform value in [0, range).  No modulo bias.
    //
    // Reference: D. Lemire, "Fast Random Integer Generation in an Interval",
    //            ACM TOMS 45(1), 2019.
    inline uint32_t bounded(uint32_t range) {
        uint64_t m = static_cast<uint64_t>(next()) * static_cast<uint64_t>(range);
        uint32_t l = static_cast<uint32_t>(m);
        if (l < range) {
            uint32_t t = (-range) % range;  // rejection threshold
            while (l < t) {
                m = static_cast<uint64_t>(next()) * static_cast<uint64_t>(range);
                l = static_cast<uint32_t>(m);
            }
        }
        return static_cast<uint32_t>(m >> 32u);
    }
};

// ---------------------------------------------------------------------------
// materialize_mprw — the hot function.
//
// Args
// ----
//   target_nodes   : [n_target] int64 — IDs of source/target nodes.
//   csr_offsets    : vector of [n_src_i + 1] int64 tensors — row pointers per
//                    edge type in the metapath.
//   csr_destinations: vector of [nnz_i] int64 tensors — column indices per
//                     edge type, sorted by source node (CSR order).
//   k              : number of random walks per target node.
//   min_visits     : minimum walk hits for an edge to be included.
//   seed           : RNG seed for reproducibility.
//
// Returns
// -------
//   edge_index : [2, num_edges] int64 — coalesced, undirected, WITH self-loops.
// ---------------------------------------------------------------------------
torch::Tensor materialize_mprw(
    torch::Tensor          target_nodes,
    std::vector<torch::Tensor> csr_offsets,
    std::vector<torch::Tensor> csr_destinations,
    int64_t                k,
    int64_t                min_visits,
    int64_t                seed
) {
    // -- Validate inputs ----------------------------------------------------
    TORCH_CHECK(target_nodes.is_contiguous() && target_nodes.dtype() == torch::kInt64,
                "target_nodes must be contiguous int64");
    const int64_t n_hops = static_cast<int64_t>(csr_offsets.size());
    TORCH_CHECK(n_hops > 0, "metapath must have at least one hop");
    TORCH_CHECK(static_cast<int64_t>(csr_destinations.size()) == n_hops,
                "csr_offsets and csr_destinations must have the same length");
    TORCH_CHECK(k >= 1, "k must be >= 1");
    TORCH_CHECK(min_visits >= 1, "min_visits must be >= 1");

    // -- Raw data pointers --------------------------------------------------
    // All CSR arrays are accessed via raw int64_t* for speed — no ATen
    // indexing in the inner loop.
    const int64_t  n_target   = target_nodes.size(0);
    const int64_t* target_ptr = target_nodes.data_ptr<int64_t>();

    // Cache raw pointers and sizes for each hop's CSR.
    struct HopCSR {
        const int64_t* offsets;      // [n_src + 1]
        const int64_t* destinations; // [nnz]
        int64_t        n_src;        // number of source nodes for this edge type
    };
    std::vector<HopCSR> hops(n_hops);
    for (int64_t h = 0; h < n_hops; ++h) {
        TORCH_CHECK(csr_offsets[h].is_contiguous() && csr_offsets[h].dtype() == torch::kInt64,
                    "csr_offsets[", h, "] must be contiguous int64");
        TORCH_CHECK(csr_destinations[h].is_contiguous() && csr_destinations[h].dtype() == torch::kInt64,
                    "csr_destinations[", h, "] must be contiguous int64");
        hops[h].offsets      = csr_offsets[h].data_ptr<int64_t>();
        hops[h].destinations = csr_destinations[h].data_ptr<int64_t>();
        hops[h].n_src        = csr_offsets[h].size(0) - 1;  // offsets has n_src+1 elements
    }

    // -- Pre-allocate work buffers ------------------------------------------
    // terminal_nodes: reused per target node (holds the k walk endpoints).
    // out_src / out_dst: grow as edges are discovered.  Reserve a generous
    // estimate to avoid repeated reallocation.
    std::vector<int64_t> terminal_nodes(k);
    std::vector<int64_t> out_src;
    std::vector<int64_t> out_dst;
    out_src.reserve(n_target * 8);   // heuristic: ~8 neighbours per node
    out_dst.reserve(n_target * 8);

    // -- PRNG ---------------------------------------------------------------
    PCG32 rng;
    rng.seed(static_cast<uint64_t>(seed));

    // -----------------------------------------------------------------------
    // MAIN LOOP — de-vectorised, cache-local.
    //
    //   for each target node u:
    //       for each walk i in [0, k):
    //           walk the full metapath from u → terminal
    //       sort terminal_nodes
    //       deduplicate + min_visits filter
    //       append surviving edges to out_src / out_dst
    // -----------------------------------------------------------------------
    for (int64_t t = 0; t < n_target; ++t) {
        const int64_t u = target_ptr[t];

        // -- Launch k walks from u ------------------------------------------
        for (int64_t wi = 0; wi < k; ++wi) {
            int64_t current = u;
            bool alive = true;

            // Walk every hop in the metapath.
            for (int64_t h = 0; h < n_hops; ++h) {
                // Bounds-check: if current is out of range for this edge type's
                // source space, the walker is dead (type mismatch / dead end).
                if (current < 0 || current >= hops[h].n_src) {
                    alive = false;
                    break;
                }

                // CSR lookup: degree = offsets[current+1] - offsets[current].
                const int64_t row_start = hops[h].offsets[current];
                const int64_t row_end   = hops[h].offsets[current + 1];
                const int64_t degree    = row_end - row_start;

                if (degree == 0) {
                    // Dead end — abort this walk.
                    alive = false;
                    break;
                }

                // Pick a uniform random neighbour using Lemire's method.
                uint32_t choice = rng.bounded(static_cast<uint32_t>(degree));
                current = hops[h].destinations[row_start + choice];
            }

            // Store terminal node.  Dead walkers get -1 (filtered later).
            terminal_nodes[wi] = alive ? current : -1;
        }

        // -- Deduplication + min_visits filtering ---------------------------
        // Sort the terminal_nodes buffer so duplicates are adjacent.
        std::sort(terminal_nodes.begin(), terminal_nodes.end());

        // Linear scan: count consecutive duplicates.
        int64_t i = 0;
        while (i < k) {
            const int64_t v = terminal_nodes[i];

            // Skip invalid walkers (marked -1, will be sorted to the front
            // since -1 < any valid node ID).
            if (v < 0) {
                ++i;
                continue;
            }

            // Skip self-loops.
            if (v == u) {
                // Advance past all copies of u.
                while (i < k && terminal_nodes[i] == u) ++i;
                continue;
            }

            // Count how many walks landed on v.
            int64_t count = 0;
            int64_t j = i;
            while (j < k && terminal_nodes[j] == v) {
                ++count;
                ++j;
            }

            // Accept edge if visit count meets threshold.
            if (count >= min_visits) {
                out_src.push_back(u);
                out_dst.push_back(v);
            }

            i = j;  // advance past this run
        }
    }

    // -----------------------------------------------------------------------
    // Post-processing: make undirected + add self-loops + coalesce.
    //
    // For every (u, v) edge, we also emit (v, u).  Then append (u, u) for
    // every target node.  Finally sort and deduplicate.
    // -----------------------------------------------------------------------
    const int64_t n_directed = static_cast<int64_t>(out_src.size());

    // Worst-case: 2 * n_directed (forward + reverse) + n_target (self-loops).
    std::vector<int64_t> all_src;
    std::vector<int64_t> all_dst;
    all_src.reserve(2 * n_directed + n_target);
    all_dst.reserve(2 * n_directed + n_target);

    // Forward edges.
    all_src.insert(all_src.end(), out_src.begin(), out_src.end());
    all_dst.insert(all_dst.end(), out_dst.begin(), out_dst.end());

    // Reverse edges (make undirected).
    all_src.insert(all_src.end(), out_dst.begin(), out_dst.end());
    all_dst.insert(all_dst.end(), out_src.begin(), out_src.end());

    // Self-loops for every target node.
    for (int64_t t = 0; t < n_target; ++t) {
        all_src.push_back(target_ptr[t]);
        all_dst.push_back(target_ptr[t]);
    }

    // -- Coalesce: sort edges lexicographically and remove duplicates -------
    const int64_t n_total = static_cast<int64_t>(all_src.size());

    // Build an index array and sort by (src, dst).
    std::vector<int64_t> perm(n_total);
    for (int64_t i = 0; i < n_total; ++i) perm[i] = i;
    std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
        if (all_src[a] != all_src[b]) return all_src[a] < all_src[b];
        return all_dst[a] < all_dst[b];
    });

    // Compact unique edges.
    std::vector<int64_t> final_src;
    std::vector<int64_t> final_dst;
    final_src.reserve(n_total);
    final_dst.reserve(n_total);

    for (int64_t i = 0; i < n_total; ++i) {
        int64_t idx = perm[i];
        int64_t s = all_src[idx];
        int64_t d = all_dst[idx];
        // Skip duplicates.
        if (!final_src.empty() && final_src.back() == s && final_dst.back() == d) {
            continue;
        }
        final_src.push_back(s);
        final_dst.push_back(d);
    }

    // -- Build output tensor ------------------------------------------------
    const int64_t n_edges = static_cast<int64_t>(final_src.size());
    torch::Tensor edge_index = torch::empty({2, n_edges}, torch::kInt64);

    // Direct memcpy from vectors into tensor storage — no element-wise copy.
    std::memcpy(edge_index.data_ptr<int64_t>(),
                final_src.data(),
                n_edges * sizeof(int64_t));
    std::memcpy(edge_index.data_ptr<int64_t>() + n_edges,
                final_dst.data(),
                n_edges * sizeof(int64_t));

    return edge_index;
}

// ---------------------------------------------------------------------------
// pybind11 module registration.
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "materialize_mprw",
        &materialize_mprw,
        "Cache-optimised MPRW materialisation (single-threaded, PCG32, Lemire).\n"
        "\n"
        "Args:\n"
        "    target_nodes (Tensor[int64]): node IDs to walk from.\n"
        "    csr_offsets (list[Tensor[int64]]): CSR row pointers per hop.\n"
        "    csr_destinations (list[Tensor[int64]]): CSR column indices per hop.\n"
        "    k (int): walks per target node.\n"
        "    min_visits (int): minimum hits for edge inclusion.\n"
        "    seed (int): RNG seed.\n"
        "\n"
        "Returns:\n"
        "    Tensor[int64] of shape [2, E]: coalesced undirected edge_index with self-loops.",
        py::arg("target_nodes"),
        py::arg("csr_offsets"),
        py::arg("csr_destinations"),
        py::arg("k"),
        py::arg("min_visits"),
        py::arg("seed")
    );
}
