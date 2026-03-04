# Paper Reproduction TODO

Status key: ✅ Done · 🔧 Needs C++ mod · 🐍 Python only · ⚠️ External dependency

---

## TABLE III — Matching Graph Statistics (|E*| and ρ*)

**Status: ✅ Fully implemented**

The `hg_stats` command already emits `RAW_EDGES_E*:` per query node and `~dens:`
as a global average. The original TODO's claim that a C++ modification was needed
is incorrect — the tag already exists in the binary.

`test_table3.py` parses both values correctly.

**Run:**
```bash
python scripts/test_table3.py HGB_ACM   --metapath "<metapath>"
python scripts/test_table3.py HGB_DBLP  --metapath "<metapath>"
python scripts/test_table3.py HGB_IMDB  --metapath "<metapath>"
# ... repeat for all 5 datasets
```

**Post-processing:** Results are printed directly. Collect into a LaTeX table.

---

## TABLE IV — Efficiency (Running Times + Speedup)

**Status: ✅ Fully implemented**

`test_table4.py` runs the full pipeline: ground truth → all six approximate
methods → summary table with `~time_per_rule:` for every method.

The `matching_graph_time` command is wired in `GraphPrepRunner.run_scale()`.

**Missing baseline:** `BoolAP` / `BoolAP+` are not in this repo. Those two
columns require the authors' separate Boolean Matrix Multiplication codebase.

**Run:**
```bash
python scripts/test_table4.py HGB_ACM --metapath "<metapath>" --topr 0.05
# Repeat for all 5 datasets
```

**Post-processing:** `~time_per_rule:` values come out of `GloResult.avg_time_s`
and `PerResult.avg_time_s`. Speedup = ExactD_time / GloD_time etc.
For T(G*), call `runner.run_scale("matching_graph_time", dataset)` separately
and read `ScaleResult.avg_time_s`.

---

## FIGURE 3 — Running Times vs Meta-path Length L (Yelp)

**Status: 🔧 Needs C++ modification**

`lensplit` already splits the master `.dat` into `.dat1`, `.dat2`, `.dat3`.
The problem is that `Effective_prop_opt_global_cross` (and the personalized
variant) hardcode loading `<dataset>-cod-global-rules.dat` — there is no way to
pass `.dat1` / `.dat2` / `.dat3` as input without modifying the C++.

**Required C++ change (effectiveness.cpp):**
In `Effective_prop_opt_global_cross` and `Effective_prop_opt_personalized_cross`,
the line that opens the rule file currently hardcodes the `.dat` suffix.
Add an optional `string rules_suffix = ".dat"` parameter to both functions,
then thread it through `main.cpp` so CLI can pass `".dat1"` etc.

Once patched, `GraphPrepRunner` will need a `suffix` parameter on `run_glo()`,
`run_per()`, and `run_scale()` — a 3-line change.

**Run (after C++ patch):**
```bash
./bin/graph_prep lensplit Yelp   # produces .dat1 .dat2 .dat3

# Then in Python, for each length in {1, 2, 3}:
runner.run_ground_truth("Yelp", topr="0.05")
runner.run_glo("GloD", "Yelp", topr="0.05", k=32, suffix=".dat1")
# ... etc.
```

**Post-processing:** Exact methods will time out for L=2 and L=3. Wrap in
`subprocess.run(timeout=3600)` — `GraphPrepRunner._run()` already has a
`timeout` parameter, default is 1200s, increase to 3600 for this experiment.

---

## FIGURE 4 — Sensitivity to |E*| (Scatter Plots)

**Status: ✅ Fully implemented**

The original TODO said per-rule timing and edge counts were missing and needed
C++ modification. This is incorrect — `SCATTER_DATA: <rule_idx>,<edges>,<time_s>`
is already emitted inside the rule loop by the current binary.

`test_figure4.py` captures this via `GloResult.scatter_data` and
`PerResult.scatter_data` (list of raw CSV strings).

**Run:**
```bash
python scripts/test_figure4.py HGB_ACM --metapath "<metapath>"
# Repeat for all datasets
```

**Post-processing (Python only 🐍):**
```python
from scipy.stats import linregress
import matplotlib.pyplot as plt

# scatter_data entries are "rule_idx,edges,time_s" strings
points = [(float(e), float(t)) for _, e, t in (s.split(",") for s in result.scatter_data)]
edges, times = zip(*points)
slope, intercept, r, *_ = linregress(edges, times)
r_squared = r ** 2

plt.scatter(edges, times, alpha=0.5)
plt.plot(sorted(edges), [slope*x + intercept for x in sorted(edges)], label=f"R²={r_squared:.3f}")
```

---

## FIGURES 5 & 6 — Effectiveness vs λ and k

**Status: ✅ Fully implemented**

`test_figure5_6.py` runs both sweeps end-to-end:
- Experiment 1: λ ∈ {0.02, 0.03, 0.04, 0.05}, fixed k
- Experiment 2: k ∈ {2, 4, 8, 16, 32}, fixed λ=0.05

Ground truth is regenerated for each λ inside the loop. topr canonicalization
prevents the "0.02 vs 0.020" silent filename mismatch.

**Run:**
```bash
python scripts/test_figure5_6.py HGB_ACM --metapath "<metapath>"
```

**Post-processing (Python only 🐍):**
Output is raw CSV printed to stdout. Pipe to a file and plot with `matplotlib`:
```bash
python scripts/test_figure5_6.py HGB_ACM --metapath "<metapath>" > results/fig5_ACM.csv
```

---

## FIGURE 7 — Approximation Ratio ε vs k

**Status: 🔧 Needs C++ modification (non-trivial)**

The binary currently computes F1 set-intersection scores but not the theoretical
approximation ratio ε = max_v |c_est(v) - c_exact(v)| / c_exact(v).

Both the exact centrality array and the estimated array exist in memory
simultaneously inside `Effective_prop_opt_global_cross`, but ε is never computed.

**Required C++ change (cod.cpp or effectiveness.cpp):**
1. After the existing F1 computation block, iterate over the `ractive` peer set.
2. For each peer v: compute `abs(c_est[v] - c_exact[v]) / c_exact[v]`.
3. Take the max over all v.
4. Emit: `std::cout << "~epsilon: " << epsilon_val << "\n";`

Once emitted, add `~epsilon:` parsing to `GloResult` and update
`GraphPrepRunner._parse_float` to extract it.

**Run (after C++ patch) — 10 seeds per (dataset, k) combination:**
```bash
# GraphPrepRunner passes K_AND_SEED as a single int, so seed=k by design.
# To get independent seeds you need separate k values, or patch the binary
# to accept argv[6] as an independent seed override (see README §2 gotcha #1).
for k in 2 4 8 16 32:
    for seed in range(10):
        runner.run_glo("GloD", dataset, topr="0.05", k=k)
        # NOTE: currently seed is locked to k — you cannot vary them independently
        # without a C++ patch to decouple argv[5] into separate K and SEED args
```

**Post-processing (Python only 🐍):** `plt.fill_between()` over mean ± std of
the 10 ε values per k.

---

## FIGURE 8 — Influence Analysis (RIS comparison)

**Status: ⚠️ External dependency — significant work**

The C++ binary does not simulate graph diffusion. Two separate pieces are needed:

**Step 1 — C++ patch:** Modify `ExactD`, `ExactD+`, `GloD` etc. to write the
actual top-R node IDs to a text file instead of just counting set intersections
in memory. Currently the `(` / `)` lines in stdout serve this purpose for `.res`
files — you may be able to reuse that output directly.

**Step 2 — Python simulation:**
```python
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import networkx as nx

# Load graph, set top-R nodes as seeds, run IC/LT cascade model
```

This figure is the most isolated from the rest of the codebase.
Recommend tackling it last.

---

## FIGURE 9 — Scalability on BPM Synthetic Dataset

**Status: ⚠️ External dependency — dataset generation required**

**What exists:** `runner.run_scale("scale", dataset, k=32)` works correctly and
returns `avg_time_s`. Memory tracking is not implemented in the binary.

**What's missing:**
1. The BPM graph generator script (not in this repo) to create synthetic datasets
   at 600K → 19.2M nodes.
2. Memory profiling wrapper.

**Run (once you have BPM datasets):**
```bash
# Memory via /usr/bin/time -v (Linux only)
/usr/bin/time -v ./bin/graph_prep scale BPM_600K 0.05 0 32 2>&1 | grep "Maximum resident"
```

Or from Python using `resource` module:
```python
import subprocess, re
out = subprocess.run(["/usr/bin/time", "-v", "./bin/graph_prep", "scale", dataset, ...],
                    capture_output=True, text=True)
mem_kb = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", out.stderr).group(1))
mem_gb = mem_kb / 1024 / 1024
```

---

## Reproduction Priority Order

Given current status, suggested order to minimize blocked time:

1. **Tables III & IV** — run immediately, no C++ changes needed
2. **Figures 5 & 6** — run immediately, no C++ changes needed
3. **Figure 4** — run immediately, no C++ changes needed
4. **Figure 3** — small C++ patch, then run
5. **Figure 7** — medium C++ patch (decouple K/SEED + add ε output)
6. **Figure 9** — blocked on BPM generator
7. **Figure 8** — most work, tackle last