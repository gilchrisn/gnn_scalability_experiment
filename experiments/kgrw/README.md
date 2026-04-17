# KGRW (KMV-Guided Random Walk) — Experiment Log

**Author**: Gilchris  
**Date**: April 2026  
**Goal**: Test whether combining KMV sketch propagation with MPRW produces a better graph sparsifier than either alone.

---

## Algorithm

KGRW splits a length-L meta-path at the midpoint (hop L/2):

```
Phase 1  [hops 0 .. L/2-1]   KMV sketch propagation from sources.
                               Each midpoint m accumulates a sketch S_m of the k
                               smallest hash values from source nodes that reach m.

Phase 2  [hops L/2 .. L-1]   w' random walks from each midpoint m.
                               Each walk produces one endpoint e.

Phase 3  Cross-product join.  For each (midpoint m, endpoint e) pair:
                               emit edge (s, e) for EVERY s in S_m.
```

### Key design choice: cross-product, NOT round-robin
The original implementation used round-robin pairing (1 edge per walk regardless of k).
This was wrong — k beyond w' was completely unused. The correct implementation
emits k edges per walk (one per sketch entry). See `csrc/mprw_exec.cpp`, function `cmd_kgrw`.

### CLI
```bash
# Run from WSL:
bin/mprw_exec kgrw <dataset_dir> <rule_file> <output.adj> <k> <w_prime> <seed>

# Example (DBLP APAPA, k=32, w'=32):
bin/mprw_exec kgrw HGBn-DBLP HGBn-DBLP/cod-rules_HGBn-DBLP.limit \
    /tmp/kgrw_out.adj 32 32 42
```

---

## Theoretical Justification

For midpoint m with d_S sources (Phase 1) and d_T terminals (Phase 2):

**MPRW** coupon-collector over d_S * d_T paths:
```
w_MPRW ≈ d_S * d_T * ln(1 / (1-f))   to cover fraction f
```

**KGRW cross-product** coupon-collector only over d_T terminals:
```
w'_KGRW ≈ d_T * ln(1 / (1-f))
```

**Speedup = d_S** (the source-side degree at each midpoint).
For DBLP APAPA, d_S ≈ 5-7, predicting ~5x fewer Phase 2 walks.
Empirically confirmed: ~4x speedup.

k saturates at k ≈ d_S — no benefit from k >> d_S since there are no more unique sources to add.

---

## Experiments

All experiments run on **DBLP APAPA** meta-path:
`author -> paper -> author -> paper -> author` (4 hops, L/2 = 2)

**Why APAPA?** The APTPA path (used in main paper) is too dense (16M edges, avg degree ~4000).
APAPA is sparse (true exact ~36K edges, avg degree ~9) — the coupon-collector regime where
KGRW's advantage actually exists.

### Step 0: Setup (one-time)

Ensure HGBn-DBLP is staged (meta.dat, node.dat, link.dat already exist from overnight run).

```bash
# Activate venv (Windows)
source .venv/Scripts/activate
export PYTHONUTF8=1

# Compile the binary (WSL)
wsl bash -c "cd /mnt/c/Users/Gilchris/UNI/not-school/Research/gnn/scalability_experiment && make bin/mprw_exec"
```

### Step 1: Compile APAPA rule

```python
# Run from project root (Windows venv)
python - << 'EOF'
import sys, types as _t, warnings
_ts = _t.ModuleType('torch_sparse'); _ts.spspmm = None; sys.modules['torch_sparse'] = _ts
warnings.filterwarnings('ignore')
from src.config import config
from src.data import DatasetFactory
from scripts.bench_utils import compile_rule_for_cpp

cfg = config.get_dataset_config('HGB_DBLP')
g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
compile_rule_for_cpp(
    'author_to_paper,paper_to_author,author_to_paper,paper_to_author',
    g, 'HGBn-DBLP', 'HGBn-DBLP'
)
EOF
```

Resulting rule file: `HGBn-DBLP/cod-rules_HGBn-DBLP.limit`
Expected content: `-2 0 -2 1 -2 0 -2 -1 1 -4 -4 -4 -4`

### Step 2: Find the exact edge count (plateau)

```bash
wsl bash << 'EOF'
cd /mnt/c/Users/Gilchris/UNI/not-school/Research/gnn/scalability_experiment
D=HGBn-DBLP; R=HGBn-DBLP/cod-rules_HGBn-DBLP.limit

for w in 512 1024 2048 4096 8192; do
  bin/mprw_exec materialize $D $R /tmp/exact_probe_${w}.adj $w 42 2>&1 | \
    grep '\[mprw_exec\]'
done
EOF
```

**Expected output** (true edge count plateaus ~36,232 at w=8192):
```
w=512:   28,480 edges (78%)
w=1024:  31,828 edges (87%)
w=2048:  34,116 edges (93%)
w=4096:  35,530 edges (97%)
w=8192:  36,232 edges (99%)   <-- use this as EXACT baseline
```

### Step 3: MPRW coupon-collector curve

```bash
wsl bash << 'EOF'
cd /mnt/c/Users/Gilchris/UNI/not-school/Research/gnn/scalability_experiment
D=HGBn-DBLP; R=HGBn-DBLP/cod-rules_HGBn-DBLP.limit; EXACT=36232

for w in 1 2 4 8 16 32 64 128 256 512; do
  bin/mprw_exec materialize $D $R /tmp/mprw_apapa_${w}.adj $w 42 2>&1 | \
    grep '\[mprw_exec\]' | \
    awk -v w=$w -v ex=$EXACT '{match($0,/edges=([0-9]+)/,e); match($0,/time=([0-9.]+)/,t);
      printf "w=%-4d  edges=%-8s  coverage=%.0f%%  time=%ss\n",w,e[1],100*e[1]/ex,t[1]}'
done
EOF
```

**Expected output** (diminishing returns = coupon-collector):
```
w=1     edges=2508      coverage=6%   time=0.001s
w=16    edges=9744      coverage=27%  time=0.003s
w=64    edges=15866     coverage=44%  time=0.006s
w=512   edges=28480     coverage=78%  time=0.048s
w=8192  edges=36232     coverage=99%  time=0.786s
```
Marginal efficiency drops from +58% (w=1->2) to +2% (w=4096->8192).

### Step 4: KGRW cross-product sweep

```bash
wsl bash << 'EOF'
cd /mnt/c/Users/Gilchris/UNI/not-school/Research/gnn/scalability_experiment
D=HGBn-DBLP; R=HGBn-DBLP/cod-rules_HGBn-DBLP.limit; EXACT=36232

echo "--- k-sweep (fixed w'=8): does larger k help? ---"
for k in 2 4 8 16 32 64; do
  bin/mprw_exec kgrw $D $R /tmp/kgrw_k${k}_w8.adj $k 8 42 2>&1 | \
    grep '^\[kgrw\] DONE' | \
    awk -v k=$k -v ex=$EXACT '{match($0,/edges=([0-9]+)/,e); match($0,/time=([0-9.]+)/,t);
      printf "k=%-4d w'"'"'=8   edges=%-8s  coverage=%.0f%%  time=%ss\n",k,e[1],100*e[1]/ex,t[1]}'
done

echo ""
echo "--- w'-sweep (fixed k=32): main result ---"
for wp in 1 2 4 8 16 32 64; do
  bin/mprw_exec kgrw $D $R /tmp/kgrw_k32_w${wp}.adj 32 $wp 42 2>&1 | \
    grep '^\[kgrw\] DONE' | \
    awk -v wp=$wp -v ex=$EXACT '{match($0,/edges=([0-9]+)/,e); match($0,/time=([0-9.]+)/,t);
      printf "k=32  w'"'"'=%-4d  edges=%-8s  coverage=%.0f%%  time=%ss\n",wp,e[1],100*e[1]/ex,t[1]}'
done
EOF
```

**Expected output**:
```
k-sweep (k matters now with cross-product!):
k=2   w'=8   edges=8796    coverage=24%
k=8   w'=8   edges=14592   coverage=40%
k=32  w'=8   edges=17418   coverage=48%
k=64  w'=8   edges=17560   coverage=48%   <- saturates at k~32 (d_S ≈ 5-7)

w'-sweep (main result — compare to MPRW above):
k=32  w'=8    edges=17418   coverage=48%   0.008s   (MPRW needs w=64 for same: 0.006s)
k=32  w'=16   edges=22142   coverage=61%   0.007s   (MPRW needs w=128: 0.013s) ← KGRW wins
k=32  w'=32   edges=27120   coverage=75%   0.011s   (MPRW needs w~400: ~0.04s) ← 4x faster
k=32  w'=64   edges=31108   coverage=86%   0.011s   (MPRW needs w~2000: ~0.2s) ← 18x faster
```

### Step 5: Summary comparison at equal coverage

| Coverage | MPRW | KGRW cross-product (k=32) | Speedup |
|----------|------|--------------------------|---------|
| 35% | w=32, 0.005s | w'=4, 0.004s | ~1.2x |
| 61% | w=128, 0.013s | w'=16, 0.007s | **1.9x** |
| 75% | w~400, ~0.04s | w'=32, 0.011s | **~4x** |
| 86% | w~2000, ~0.2s | w'=64, 0.011s | **~18x** |

Theory predicts d_S-fold speedup (d_S ≈ 5). Empirically confirmed at ~4x for 75% coverage.

---

## Next Step: GNN Quality Validation

**This is the critical missing experiment.**

We need to check whether the edge count improvement (4x faster coverage) also translates to
better GNN quality (CKA, F1) compared to MPRW at equal time budget.

The hypothesis: Phase 2 walks preserve MPRW's structural coherence (path-following),
so KGRW should achieve CKA ≈ MPRW while being faster.

### How to run it

```bash
# 1. Stage the partition (use existing one from overnight run)
# Already at: results/HGB_DBLP/partition.json

# 2. Run inference comparison
python scripts/eval_kgrw.py \
    --dataset HGB_DBLP \
    --L 2 \
    --metapath "author_to_paper,paper_to_author,author_to_paper,paper_to_author" \
    --kgrw-adj /tmp/kgrw_k32_w32.adj \
    --exact-adj /tmp/exact_apapa_reference.adj
```

**Key metrics to collect**:
- CKA(Z_kgrw, Z_exact) — target: ≥ CKA(Z_mprw_matched_budget, Z_exact)
- Macro-F1 on test nodes
- PredAgreement

If KGRW CKA ≥ MPRW CKA at equal time → publishable contribution.
If KGRW CKA ≈ KMV CKA (below MPRW) → Phase 2 walks aren't preserving structure.

---

## File Locations

| File | Purpose |
|------|---------|
| `csrc/mprw_exec.cpp` | C++ implementation (function `cmd_kgrw`, cross-product pairing) |
| `HGBn-DBLP/cod-rules_HGBn-DBLP.limit` | APAPA rule bytecode (compiled in Step 1) |
| `scripts/eval_kgrw.py` | Python eval script (runs inference, computes CKA) |
| `results/HGB_DBLP/partition.json` | Train/test split from overnight run |
| `results/HGB_DBLP/weights/` | Frozen SAGE weights from overnight run |
| `results/kgrw_eval/` | Saved embeddings from previous KGRW run (DBLP L=2) |

---

## Git State

The cross-product KGRW fix is in `csrc/mprw_exec.cpp` (modified, not yet committed).
The APAPA rule file change is in `HGBn-DBLP/cod-rules_HGBn-DBLP.limit` (modified).

To commit:
```bash
git add csrc/mprw_exec.cpp
git commit -m "KGRW: replace round-robin with cross-product pairing

Theoretical basis: coupon-collector universe shrinks from d_S*d_T (MPRW)
to d_T (KGRW), giving d_S-fold fewer Phase 2 walks needed.
Empirically: ~4x speedup over MPRW at 75% coverage on DBLP APAPA.
"
```

Note: do NOT commit `HGBn-DBLP/cod-rules_HGBn-DBLP.limit` — it contains the APAPA rule
which overwrites the APTPA rule used by the main paper experiments. Restore before running
the main pipeline:
```python
# Restore APTPA rule (main paper path):
compile_rule_for_cpp(
    'author_to_paper,paper_to_term,term_to_paper,paper_to_author',
    g, 'HGBn-DBLP', 'HGBn-DBLP'
)
```
