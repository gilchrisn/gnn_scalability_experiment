#!/usr/bin/env bash
# Run full pipeline once (exp1/exp2 + exact+KMV baseline), then fixed-w MPRW
# across many seeds and aggregate MPRW rows.
#
# Workflow:
#   1) exp1_partition.py  (40/60 split)
#   2) exp2_train.py      (train on 40%)
#   3) exp3_inference.py  exact + KMV(k=8), skip MPRW
#   4) Calibrate one fixed MPRW w so edge density is within [105%, 115%] of KMV
#   5) exp3_inference.py  MPRW-only (fixed w) for N seeds
#
# Outputs:
#   results/<dataset>/mprw_k8_fixedw_sweep/
#     baseline_exact_kmv.csv
#     calibration.json
#     seed_<seed>.csv
#     mprw_seed_rows.csv
#     mprw_seed_summary.csv

set -euo pipefail

DS="HGB_DBLP"
MP="author_to_paper,paper_to_venue,venue_to_paper,paper_to_author"
TARGET_TYPE="author"
TRAIN_FRAC="0.4"
DEPTHS="2 3 4"
K="8"
EPOCHS="100"
PARTITION_SEED="42"
TRAIN_SEED="42"
START_SEED="0"
N_SEEDS="20"
CALIB_SEED="0"
LOW_RATIO="1.05"
HIGH_RATIO="1.15"
TARGET_RATIO="1.10"
MAX_RSS_GB="32"
TIMEOUT="600"
OUT_DIR=""

usage() {
  cat <<'EOF'
Usage: bash run_mprw_k8_fixedw_seeds.sh [options]

Options:
  --dataset <name>         Dataset key (default: HGB_DBLP)
  --metapath <csv>         Metapath edge names (default: DBLP APVPA)
  --target-type <name>     Target node type for exp1 (default: author)
  --train-frac <float>     Train fraction for exp1 (default: 0.4)
  --depths "<ints>"        Depth list for exp2/exp3 (default: "2 3 4")
  --k <int>                KMV k (default: 8)
  --epochs <int>           Exp2 epochs (default: 100)
  --partition-seed <int>   Seed for exp1 (default: 42)
  --train-seed <int>       Seed for exp2 (default: 42)
  --start-seed <int>       First seed for MPRW sweep (default: 0)
  --n-seeds <int>          Number of MPRW seeds (default: 20)
  --calib-seed <int>       Seed used during fixed-w calibration (default: 0)
  --low-ratio <float>      Min MPRW/KMV edge ratio (default: 1.05)
  --high-ratio <float>     Max MPRW/KMV edge ratio (default: 1.15)
  --target-ratio <float>   Preferred ratio inside band (default: 1.10)
  --max-rss-gb <float>     RSS guard passed to exp3 (default: 32)
  --timeout <int>          Timeout seconds passed to exp3/worker (default: 600)
  --out-dir <path>         Output directory (default: results/<dataset>/mprw_k8_fixedw_sweep)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)         DS="$2"; shift 2 ;;
    --metapath)        MP="$2"; shift 2 ;;
    --target-type)     TARGET_TYPE="$2"; shift 2 ;;
    --train-frac)      TRAIN_FRAC="$2"; shift 2 ;;
    --depths)          DEPTHS="$2"; shift 2 ;;
    --k)               K="$2"; shift 2 ;;
    --epochs)          EPOCHS="$2"; shift 2 ;;
    --partition-seed)  PARTITION_SEED="$2"; shift 2 ;;
    --train-seed)      TRAIN_SEED="$2"; shift 2 ;;
    --start-seed)      START_SEED="$2"; shift 2 ;;
    --n-seeds)         N_SEEDS="$2"; shift 2 ;;
    --calib-seed)      CALIB_SEED="$2"; shift 2 ;;
    --low-ratio)       LOW_RATIO="$2"; shift 2 ;;
    --high-ratio)      HIGH_RATIO="$2"; shift 2 ;;
    --target-ratio)    TARGET_RATIO="$2"; shift 2 ;;
    --max-rss-gb)      MAX_RSS_GB="$2"; shift 2 ;;
    --timeout)         TIMEOUT="$2"; shift 2 ;;
    --out-dir)         OUT_DIR="$2"; shift 2 ;;
    -h|--help)         usage; exit 0 ;;
    *)                 echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="results/${DS}/mprw_k8_fixedw_sweep"
fi

if [[ -f ".venv/bin/activate" ]]; then
  # Linux/WSL venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f ".venv/Scripts/activate" ]]; then
  # Git-Bash on Windows
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
fi
export PYTHONUTF8=1

if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "python/python3 not found on PATH." >&2
  exit 1
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

PART_JSON="results/${DS}/partition.json"
WEIGHTS_DIR="results/${DS}/weights"
BASELINE_CSV="${OUT_DIR}/baseline_exact_kmv.csv"
CALIB_JSON="${OUT_DIR}/calibration.json"
AGG_CSV="${OUT_DIR}/mprw_seed_rows.csv"
SUMMARY_CSV="${OUT_DIR}/mprw_seed_summary.csv"

mkdir -p "$OUT_DIR"
rm -f "$BASELINE_CSV" "$CALIB_JSON" "$AGG_CSV" "$SUMMARY_CSV"
TEST_FRAC=$("$PY" -c "v=float('$TRAIN_FRAC'); print(f'{1.0-v:.3f}')")

log "================================================================"
log "Dataset=${DS}"
log "Metapath=${MP}"
log "Split=train:${TRAIN_FRAC} test:${TEST_FRAC}"
log "Depths=${DEPTHS}  k=${K}  epochs=${EPOCHS}"
log "Seeds start=${START_SEED} n=${N_SEEDS} calib_seed=${CALIB_SEED}"
log "Density target band=[${LOW_RATIO}, ${HIGH_RATIO}] target=${TARGET_RATIO}"
log "Out=${OUT_DIR}"
log "================================================================"

# ---------------------------------------------------------------------------
# 1) Partition 40/60
# ---------------------------------------------------------------------------
log "=== EXP1: partition (${TRAIN_FRAC}/${TEST_FRAC}) ==="
"$PY" scripts/exp1_partition.py \
  --dataset "$DS" \
  --target-type "$TARGET_TYPE" \
  --train-frac "$TRAIN_FRAC" \
  --seed "$PARTITION_SEED"

[[ -f "$PART_JSON" ]] || { echo "Missing partition json: $PART_JSON" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 2) Train on 40%
# ---------------------------------------------------------------------------
log "=== EXP2: train ==="
"$PY" scripts/exp2_train.py "$DS" \
  --metapath "$MP" \
  --depth ${DEPTHS} \
  --epochs "$EPOCHS" \
  --partition-json "$PART_JSON" \
  --seed "$TRAIN_SEED"

# ---------------------------------------------------------------------------
# 3) Baseline: Exact + KMV(k=8), skip MPRW
# ---------------------------------------------------------------------------
log "=== EXP3 baseline: Exact + KMV(k=${K}) ==="
"$PY" scripts/exp3_inference.py "$DS" \
  --metapath "$MP" \
  --depth ${DEPTHS} \
  --k-values "$K" \
  --partition-json "$PART_JSON" \
  --weights-dir "$WEIGHTS_DIR" \
  --hash-seed "$CALIB_SEED" \
  --max-rss-gb "$MAX_RSS_GB" \
  --timeout "$TIMEOUT" \
  --skip-mprw \
  --output "$BASELINE_CSV"

KMV_EDGES=$("$PY" - <<'PY' "$BASELINE_CSV" "$K"
import csv, sys
p, k = sys.argv[1], sys.argv[2]
val = None
with open(p, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r.get("Method") == "KMV" and str(r.get("k_value", "")) == str(k):
            if r.get("Edge_Count", ""):
                val = int(float(r["Edge_Count"]))
                break
if val is None:
    raise SystemExit("Could not find KMV Edge_Count in baseline CSV.")
print(val)
PY
)
log "Baseline KMV edges(k=${K}) = ${KMV_EDGES}"

# ---------------------------------------------------------------------------
# 4) Calibrate fixed w so MPRW edge ratio is in [LOW_RATIO, HIGH_RATIO]
# ---------------------------------------------------------------------------
log "=== Calibrating fixed MPRW w ==="
CALIB_W=$("$PY" - <<'PY' \
  "$DS" "$MP" "$K" "$KMV_EDGES" "$LOW_RATIO" "$HIGH_RATIO" "$TARGET_RATIO" \
  "$CALIB_SEED" "$TIMEOUT" "$CALIB_JSON" "$PWD"
import json, math, os, subprocess, sys
from pathlib import Path

import torch

dataset      = sys.argv[1]
metapath     = sys.argv[2]
k            = int(sys.argv[3])
kmv_edges    = int(sys.argv[4])
low_ratio    = float(sys.argv[5])
high_ratio   = float(sys.argv[6])
target_ratio = float(sys.argv[7])
calib_seed   = int(sys.argv[8])
timeout_s    = int(sys.argv[9])
calib_json   = Path(sys.argv[10])
project_root = Path(sys.argv[11]).resolve()

sys.path.insert(0, str(project_root))
from src.config import config
from src.data import DatasetFactory
from src.kernels.mprw import parse_metapath_triples

cfg      = config.get_dataset_config(dataset)
folder   = config.get_folder_name(dataset)
data_dir = project_root / folder

g_full, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
triples = parse_metapath_triples(metapath, g_full)

mprw_work_dir = data_dir / "mprw_work"
mprw_work_dir.mkdir(parents=True, exist_ok=True)
meta_file = mprw_work_dir / "meta.json"

meta = {"n_target": g_full[cfg.target_node].num_nodes, "target_type": cfg.target_node, "steps": []}
for i, (src_t, edge_name, dst_t) in enumerate(triples):
    edge_file = mprw_work_dir / f"edge_{i}.pt"
    torch.save(g_full[src_t, edge_name, dst_t].edge_index.cpu(), edge_file)
    meta["steps"].append({
        "src_type": src_t, "edge_name": edge_name, "dst_type": dst_t,
        "n_src": g_full[src_t].num_nodes, "n_dst": g_full[dst_t].num_nodes,
        "edge_file": str(edge_file),
    })
with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(meta, f)

mprw_worker = project_root / "scripts" / "mprw_worker.py"

n_target = int(meta["n_target"])
denom = max(1, n_target * (n_target - 1))

def density_from_edges(edges: int) -> float:
    real_edges = max(0, int(edges) - n_target)
    return real_edges / float(denom)

kmv_density = density_from_edges(kmv_edges)
low_density = kmv_density * low_ratio
high_density = kmv_density * high_ratio
target_density = kmv_density * target_ratio

low_edges    = n_target + math.ceil(low_density * denom)
high_edges   = n_target + math.floor(high_density * denom)
target_edges = n_target + round(target_density * denom)

cache = {}
def run_w(w: int):
    w = max(1, int(w))
    if w in cache:
        return cache[w]
    out_pt = mprw_work_dir / f"_calib_w_{w}.pt"
    cmd = [sys.executable, str(mprw_worker), str(meta_file), str(out_pt), str(w), str(calib_seed)]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_s)
    ei = torch.load(out_pt, weights_only=True)
    try:
        out_pt.unlink()
    except OSError:
        pass
    peak_ram_mb = 0.0
    for line in res.stdout.splitlines():
        if line.strip().lower().startswith("peak_ram_mb:"):
            try:
                peak_ram_mb = float(line.split(":", 1)[1])
            except ValueError:
                pass
            break
    edges = int(ei.size(1))
    density = density_from_edges(edges)
    if kmv_density > 0:
        density_ratio = density / kmv_density
    else:
        density_ratio = float(edges) / float(max(1, kmv_edges))
    val = {
        "w": w,
        "edges": edges,
        "density": density,
        "density_ratio": density_ratio,
        "peak_ram_mb": peak_ram_mb,
    }
    cache[w] = val
    return val

# Exponential search for lower bound hitting low_edges
w_hi = 1
v_hi = run_w(w_hi)
MAX_W = 1 << 20
while v_hi["edges"] < low_edges and w_hi < MAX_W:
    w_hi *= 2
    v_hi = run_w(w_hi)

if v_hi["edges"] < low_edges:
    # Saturated before hitting low band: pick closest to target from sampled points.
    candidates = list(cache.values())
else:
    # Binary search smallest w with edges >= low_edges
    w_lo = max(1, w_hi // 2)
    while w_lo + 1 < w_hi:
        mid = (w_lo + w_hi) // 2
        v_mid = run_w(mid)
        if v_mid["edges"] >= low_edges:
            w_hi = mid
        else:
            w_lo = mid
    # Evaluate neighborhood to hit band/target more precisely
    cands = set()
    for base in (w_hi, w_lo):
        for d in range(-4, 5):
            cands.add(max(1, base + d))
    candidates = [run_w(w) for w in sorted(cands)]

in_band = [c for c in candidates if low_density <= c["density"] <= high_density]
pool = in_band if in_band else candidates
chosen = min(pool, key=lambda c: (abs(c["density"] - target_density), c["w"]))

out = {
    "dataset": dataset,
    "metapath": metapath,
    "k": k,
    "kmv_edges": kmv_edges,
    "kmv_density": kmv_density,
    "low_ratio": low_ratio,
    "high_ratio": high_ratio,
    "target_ratio": target_ratio,
    "low_density": low_density,
    "high_density": high_density,
    "target_density": target_density,
    "low_edges": low_edges,
    "high_edges": high_edges,
    "target_edges": target_edges,
    "calib_seed": calib_seed,
    "chosen": chosen,
    "in_band": bool(low_density <= chosen["density"] <= high_density),
    "candidates": sorted(candidates, key=lambda x: x["w"]),
}
calib_json.parent.mkdir(parents=True, exist_ok=True)
with open(calib_json, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print(chosen["w"])
PY
)
log "Chosen fixed w = ${CALIB_W}  (details: ${CALIB_JSON})"

# ---------------------------------------------------------------------------
# 5) MPRW seed sweep with fixed w
# ---------------------------------------------------------------------------
log "=== MPRW fixed-w seed sweep ==="
FAILED=0
END_SEED=$((START_SEED + N_SEEDS - 1))

for SEED in $(seq "$START_SEED" "$END_SEED"); do
  SEED_CSV="${OUT_DIR}/seed_${SEED}.csv"
  rm -f "$SEED_CSV"
  log "--- seed ${SEED}/${END_SEED} ---"

  if ! "$PY" scripts/exp3_inference.py "$DS" \
      --metapath "$MP" \
      --depth ${DEPTHS} \
      --k-values "$K" \
      --partition-json "$PART_JSON" \
      --weights-dir "$WEIGHTS_DIR" \
      --skip-exact \
      --kmv-mat-only \
      --hash-seed "$SEED" \
      --mprw-fixed-w "$CALIB_W" \
      --max-rss-gb "$MAX_RSS_GB" \
      --timeout "$TIMEOUT" \
      --output "$SEED_CSV"; then
    log "WARNING: seed ${SEED} failed"
    FAILED=$((FAILED + 1))
    continue
  fi

  "$PY" - <<'PY' "$SEED_CSV" "$AGG_CSV" "$SEED" "$CALIB_W"
import csv, sys
seed_csv, agg_csv, seed, fixed_w = sys.argv[1:5]
with open(seed_csv, newline="", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r.get("Method") == "MPRW"]
if not rows:
    raise SystemExit(0)
fields = ["Seed", "FixedW"] + list(rows[0].keys())
write_header = False
try:
    write_header = (open(agg_csv, "r", encoding="utf-8").read().strip() == "")
except FileNotFoundError:
    write_header = True
with open(agg_csv, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        w.writeheader()
    for r in rows:
        out = {"Seed": seed, "FixedW": fixed_w}
        out.update(r)
        w.writerow(out)
PY
done

if [[ -f "$AGG_CSV" ]]; then
  "$PY" - <<'PY' "$AGG_CSV" "$SUMMARY_CSV"
import csv, statistics, sys
in_csv, out_csv = sys.argv[1:3]

def to_f(v):
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None

groups = {}
with open(in_csv, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        key = (r.get("k_value", ""), r.get("L", ""))
        g = groups.setdefault(key, {"seeds": set(), "f1": [], "de": [], "edges": []})
        g["seeds"].add(r.get("Seed", ""))
        for src, dst in [("Macro_F1", "f1"), ("Dirichlet_Energy", "de"), ("Edge_Count", "edges")]:
            val = to_f(r.get(src, ""))
            if val is not None:
                g[dst].append(val)

def mean(xs): return statistics.mean(xs) if xs else ""
def std(xs): return statistics.stdev(xs) if len(xs) >= 2 else (0.0 if len(xs) == 1 else "")

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    fields = ["k_value", "L", "seed_count", "macro_f1_mean", "macro_f1_std",
              "dirichlet_mean", "dirichlet_std", "edge_count_mean", "edge_count_std"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for (k, L), g in sorted(groups.items(), key=lambda x: (int(x[0][0]), int(x[0][1]))):
        w.writerow({
            "k_value": k,
            "L": L,
            "seed_count": len(g["seeds"]),
            "macro_f1_mean": mean(g["f1"]),
            "macro_f1_std": std(g["f1"]),
            "dirichlet_mean": mean(g["de"]),
            "dirichlet_std": std(g["de"]),
            "edge_count_mean": mean(g["edges"]),
            "edge_count_std": std(g["edges"]),
        })
PY
fi

log "================================================================"
log "Done."
log "Failed seeds: ${FAILED}"
log "Baseline (Exact + KMV): ${BASELINE_CSV}"
log "Calibration:            ${CALIB_JSON}"
log "MPRW seed rows:         ${AGG_CSV}"
log "MPRW summary:           ${SUMMARY_CSV}"
log "================================================================"
