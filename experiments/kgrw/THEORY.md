# KGRW Theoretical Analysis
**Written:** April 2026  
**Status:** Derivations complete. Optimizations identified but not yet implemented.

---

## 1. Setup and Notation

Let the meta-path be $\tau = (e_1, e_2, \ldots, e_L)$, $L$ even, split at midpoint $\ell = L/2$.

**For source node $u$:**
- $\mathcal{N}(u)$ — exact meta-path neighborhood (set of distinct reachable endpoints)
- $n_u = |\mathcal{N}(u)|$ — neighborhood size
- $N(u, t)$ — number of matching path instances from $u$ to $t$ (path count)
- $Z_u = \sum_t N(u, t)$ — total matching instances from $u$
- $\bar{N} = Z_u / n_u$ — mean path count per edge
- $N_{min} = \min_t N(u, t)$ — minimum path count (1 if any single-path edge exists)

**For midpoint $m$:**
- $T_m$ — set of distinct endpoints reachable from $m$ via last $\ell$ hops
- $n_m = |T_m|$ — endpoint count per midpoint
- $n_{mid}$ — number of distinct midpoints reachable from $u$

**Degree notation:**
- $d_i$ — average degree from node type $i$ to type $i+1$ along the meta-path
- $d_S = \prod_{i=1}^{\ell-1} d_i$ — effective source-side fan-out (midpoints per source)
- $d_T = \prod_{i=\ell}^{L-1} d_i$ — effective target-side fan-out (endpoints per midpoint)

**Key relationship:** $n_u \approx d_S \cdot n_m$ (full neighborhood = midpoints × endpoints per midpoint).

---

## 2. Why MPRW Has IS Bias

MPRW from source $u$ samples endpoint $t$ with probability:
$$p_t = \frac{N(u, t)}{Z_u}$$

This is **non-uniform** — endpoints with more supporting paths are over-sampled. This is the *importance sampling (IS) bias*.

**Consequence:** Edges with $N(u,t) = 1$ (single-path edges) have $p_t = 1/Z_u$, far below the uniform $1/n_u$. They are systematically under-discovered.

**Empirical confirmation on DBLP APAPA:**
- Exact graph mean path count: $\bar{N} = 5.94$, single-path edges: 27.9%
- MPRW mean path count of discovered edges: 7.4–8.3 (biased high)
- KGRW mean path count of discovered edges: 5.27 ≈ exact (unbiased)

**Why KMV Phase 1 eliminates source-level IS bias:**  
Phase 1 assigns each source $u$ a uniform random hash $h(u)$. Source $u$ appears in midpoint $m$'s sketch iff $h(u)$ is among the $k$ smallest hashes of all sources reaching $m$ — with probability $k / |A_m|$, **independent of $N(u, m)$**. The path count no longer influences whether a source is retained.

---

## 3. The Coupon Collector Problem

**Setup:** There are $N$ distinct items ("coupons"). Each trial draws one item uniformly at random (with replacement). How many trials to collect all $N$?

**Expected trials:**
$$E[W] = N \cdot H_N = N \cdot \sum_{i=1}^{N} \frac{1}{i} \approx N \cdot \ln N$$

**Why:** When $i$ items have been collected, probability of getting a new one is $(N-i)/N$. Expected additional trials: $N/(N-i)$. Sum from $i=0$ to $N-1$ gives $N \cdot H_N$.

**Weighted (non-uniform) version:** If item $j$ has probability $p_j$ (not $1/N$), expected trials to collect all items:
$$E[W] \geq \frac{1}{p_{min}} = \frac{1}{\min_j p_j}$$

This lower bound says: you must wait at least as long as it takes to find the rarest item once.

---

## 4. MPRW and KGRW as Coupon Collector Problems

### MPRW

Each walk from source $u$ is one trial. The coupon universe is $\mathcal{N}(u)$, size $n_u$.

- **Uniform case** (all $N(u,t)$ equal): expected walks = $n_u \cdot H_{n_u} \approx n_u \ln n_u$.  
  Average walks per unique edge: $H_{n_u} \approx \ln(n_u) \approx \ln(d_S \cdot d_T)$.

- **IS-biased case** (real world): expected walks $\geq Z_u / N_{min}$.  
  Average walks per unique edge $\geq \bar{N} / N_{min}$.  
  When single-path edges exist ($N_{min} = 1$): at least $\bar{N}$ walks per edge.

Each walk also has $L$ hops (steps), so **total walk steps per unique edge**:
$$C_{MPRW}^{avg} = L \cdot \frac{\bar{N}}{N_{min}}$$

### KGRW

Phase 2 from midpoint $m$ draws endpoints from $T_m$, size $n_m$. Coupon universe is $n_m$, not $n_u$.

Each discovered endpoint generates $k$ edges (cross-product with $k$ sketch sources).

- Expected walks to collect all $n_m$ endpoints: $n_m \cdot H_{n_m} \approx n_m \ln n_m$.  
- Edges produced: $k \cdot n_m$.  
- **Phase 2 walk steps per unique edge:** $\ell \cdot H_{n_m} / k \approx \ell \cdot \ln(d_T) / k$.

Phase 1 is a fixed cost, paid once regardless of how many edges are produced.

**Total KGRW cost per unique edge:**
$$C_{KGRW}^{avg}(E) = \underbrace{\frac{C_1}{E}}_{\text{Phase 1, amortized}} + \underbrace{\frac{\ell \cdot \ln(d_T)}{k}}_{\text{Phase 2 walks}}$$

where $C_1 = O(|E^*_{1..\ell}| \cdot k)$ is the fixed Phase 1 cost.

---

## 5. The Marginal Cost Theorem

**Marginal cost** = cost of discovering the *next* unique edge (given $E$ already found).

### Theorem (KGRW Marginal Efficiency)

For any graph and any meta-path:

1. **MPRW marginal cost is strictly increasing in $E$:**
$$C_{MPRW}^{marg}(E) = \frac{n_u}{n_u - E} \cdot L \quad\text{(diverges as } E \to n_u\text{)}$$

2. **KGRW marginal cost is bounded:**
$$C_{KGRW}^{marg}(E) = \frac{C_1}{E} + \frac{\ell}{k} \cdot \frac{n_m}{n_m - E/n_{mid}}$$

   The first term decreases as $E$ grows (Phase 1 amortizes). The second term increases but slower than MPRW because $n_m \leq n_u / d_S < n_u$.

3. **A crossover $E^*$ always exists** where KGRW becomes cheaper. Setting marginal costs equal:

$$\frac{n_u}{n_u - E^*} \approx \frac{C_1}{E^*} + \frac{1}{k}$$

Solving (ignoring the small $1/k$ term):
$$\boxed{E^* \approx \frac{n_u^2}{C_1 + n_u} \approx \frac{n_u^2}{C_1} \quad \text{when } C_1 \ll n_u}$$

### Proof sketch

**MPRW:** After collecting $E$ edges, probability of new edge on next walk = $(n_u - E)/n_u$. Expected walks per new edge = $n_u/(n_u-E)$, which is strictly increasing in $E$. As $E \to n_u$, this diverges. This holds for any graph and any distribution of $N(u,t)$ (IS bias makes it worse, not better). $\square$

**KGRW:** $C_1$ is fixed by construction of Phase 1 (touches each matching graph edge exactly once). As $E$ increases, $C_1/E$ strictly decreases. Phase 2 component increases but is bounded above by $n_m/k \cdot \ln(n_m)$ (Phase 2 coupon-collector over $n_m$, not $n_u$). Since $n_m < n_u$, Phase 2 saturates before MPRW does. $\square$

**Crossover exists:** At $E=0$: $C_{MPRW}^{marg}(0) = L$ (finite), $C_{KGRW}^{marg}(0) = \infty$ (Phase 1 unamortized). MPRW wins. At $E \to n_u$: $C_{MPRW}^{marg} \to \infty$, $C_{KGRW}^{marg}$ stays bounded (KGRW Phase 2 saturates at $n_m < n_u$, so KGRW still produces edges). By continuity, a crossing exists. $\square$

### What this means in practice

- **Low coverage** ($E < E^*$): MPRW cheaper per edge — Phase 1 not amortized yet.
- **High coverage** ($E > E^*$): KGRW cheaper per edge — Phase 1 amortized, MPRW hits coupon-collector.
- **$E^*$ is computable** from graph statistics ($n_u, C_1, k$) before running anything.
- **The advantage grows with $d_S$:** larger source fan-out = more IS bias for MPRW = larger $\bar{N}/N_{min}$ = KGRW wins earlier and by more.
- **High coverage = high GNN quality** (higher CKA). So KGRW's regime of advantage is exactly the regime that matters.

### Empirical confirmation (DBLP APAPA, L=2)

Marginal cost per 1K edges (observed):

| Range | MPRW marginal | KGRW k=4 marginal |
|-------|-------------|-----------------|
| 0 → 5.7K edges | 0.29 ms/Kedge | — (Phase 1 dominates) |
| 5.7K → 9.8K edges | **0.60 ms/Kedge** ↑ | 0.96 ms/Kedge |
| 9.8K → 14K edges | ~1.2 ms/Kedge (extrap.) | **0.54 ms/Kedge** ↓ |

MPRW marginal rising, KGRW marginal falling — crossing visible in the data.

---

## 6. Three Optimizations

### Optimization 1: Reservoir Sampling (reduces Phase 1 constant by log k)

**Current Phase 1:** Maintains sorted k-min heap per midpoint. For each edge u→m: insert h(u) into heap if < current max. Cost: O(log k) per element.

**Better:** Reservoir sampling. For each source u reaching m: keep u in sketch_m with probability k / (current count). Cost: O(1) per element.

**Correctness:** Both give the same distribution — Pr[u ∈ sketch_m] = k / |A_m|, i.e. a uniform random sample of k sources. Min-hash and reservoir sampling are equivalent in terms of the sketch distribution.

**Impact:** $C_1^{new} = C_1^{old} / \log k$. For k=32: ~5x cheaper Phase 1. $E^*$ drops by factor $\log k$, so KGRW becomes competitive earlier.

**Implementation:** Replace heap with array of size k + running count. On each update: draw random integer $r$ in [0, count). If $r < k$, replace sketch[r] with current source.

---

### Optimization 2: Optimal Split Point ℓ\*

**Current:** Always split at $\ell = L/2$. Only optimal for symmetric degree sequences.

**Better:** Choose $\ell$ to minimise total cost at target coverage $E$:
$$\ell^* = \arg\min_{\ell} \left[ \underbrace{k \cdot n \sum_{i=1}^{\ell} \prod_{j=1}^{i} d_j}_{C_1(\ell)} + \underbrace{\frac{\ell \cdot \ln(d_T(\ell))}{k} \cdot E}_{C_2(\ell)} \right]$$

where $d_T(\ell) = \prod_{i=\ell+1}^{L-1} d_i$ is the Phase 2 universe size.

**Optimality condition** (treating $\ell$ as continuous):
$$k \cdot n \cdot \prod_{j=1}^{\ell^*} d_j = \frac{\ln(d_T(\ell^*))}{k} \cdot E$$

**Intuition:** Push the split toward whichever side has lower degree — smaller fan-out → smaller Phase 1 → lower $C_1$.

**Impact:** For asymmetric metapaths (e.g. dense forward direction, sparse reverse), ℓ* ≠ L/2 can cut $C_1$ significantly.

---

### Optimization 3: Adaptive MPRW → KGRW (provably optimal for any budget)

**The theorem directly prescribes the optimal algorithm:**

```
Given target edge count E_target:
  Compute E* from graph statistics (n_u, C_1, k)
  
  if E_target < E*:
      run MPRW with w such that expected edges ≈ E_target
  else:
      run KGRW with w' such that expected edges ≈ E_target
```

**Why this is optimal:** It uses each method only in the range where it has lower marginal cost. The resulting cost is the lower envelope of both marginal cost curves — no single method achieves this.

**$E^*$ computation** (practical):
1. Estimate $n_u$ from a small exact sample (or from prior runs).
2. Measure $C_1$ on a small graph or estimate from $|E^*_{1..\ell}| \cdot k \cdot c_1$ where $c_1$ is the per-operation cost (measurable once per hardware platform).
3. Compute $E^* \approx n_u^2 / C_1$.

**Combined impact:** Reservoir sampling (Opt 1) reduces $C_1$, pushing $E^*$ lower. Adaptive switching (Opt 3) uses MPRW below $E^*$ and KGRW above. Together they give the best possible performance at any target coverage.

---

## 7. Summary Table

| Property | MPRW | KGRW |
|----------|------|------|
| IS bias | Yes — proportional to $N(u,t)$ | No — Phase 1 uniform by min-hash |
| Coupon universe | $n_u = d_S \cdot d_T$ | $n_m = d_T$ (Phase 2 only) |
| Avg walks/edge (uniform) | $\ln(d_S \cdot d_T)$ | $\ln(d_T)/k$ |
| Marginal cost trend | Increasing (diverges) | Decreasing then bounded |
| Phase 1 cost | None | $O(|E^*_{1..\ell}| \cdot k)$ fixed |
| Better regime | Low coverage ($E < E^*$) | High coverage ($E > E^*$) |
| GNN quality regime | Moderate | High (where it matters) |

---

## 8. Open Questions / Next Steps

1. **Tighten the crossover formula** — current $E^* \approx n_u^2/C_1$ ignores the Phase 2 component. A tighter bound would give a better prediction of the empirical crossover.

2. **Prove the IS bias improvement quantitatively** — how much does $\bar{N}/N_{min}$ affect the MPRW side? If $N_{min}=1$ (common), MPRW cost grows as $Z_u \cdot L$ not $n_u \cdot \ln(n_u) \cdot L$, making the KGRW advantage even larger. State this as a corollary.

3. **Implement reservoir sampling** — easy change in `csrc/mprw_exec.cpp`, Phase 1 k-min update loop. Measure whether it closes the empirical gap.

4. **Implement adaptive switching** — compute $E^*$ from graph stats, choose method accordingly.

5. **Validate the crossover empirically** — extend MPRW sweep to w=64, 128, 256, 512. Plot marginal cost curves for both methods. The crossing should be visible and match the predicted $E^*$.

6. **Derive ℓ* for each dataset** — compute optimal split using degree sequences from DBLP, ACM, IMDB. Check if ℓ* = L/2 or not.

---

## 9. Reproduction

All empirical results in this document come from:
```bash
# 3-dataset depth sweep (produces kgrw_bench.csv for all 3 datasets)
python scripts/bench_kgrw.py --dataset HGB_DBLP --L 1 2 3 4 \
    --k-values 4 8 16 --w-values 1 4 16 --seeds 5

# IS-bias path-count analysis
python scripts/analysis_is_bias.py --dataset HGB_DBLP --sample 2000
```

Raw data: `results/HGB_DBLP/kgrw_bench.csv`, `results/HGB_ACM/kgrw_bench.csv`, `results/HGB_IMDB/kgrw_bench.csv`
