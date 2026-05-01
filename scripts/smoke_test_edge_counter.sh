#!/usr/bin/env bash
#
# Smoke test for the unified edge counter (count_unique_undirected_edges /
# AWK_UNIQUE_UNDIR_EDGES) added to scripts/bench_utils.py.
#
# Runs a tiny slice of the fraction sweep on HGB_ACM and asserts that the
# new Edge_Count column has the right cross-method properties:
#
#   1. KMV at k=32 >= KMV at k=4  (more sketch capacity → more unique
#      neighbours captured; same-or-larger).
#   2. MPRW at w=32 >= MPRW at w=1  (more walks → more unique pairs).
#   3. MPRW at any w <= KMV at k_max × 1.10  (10% slack; symmetrisation
#      doubling bug would push MPRW well above KMV's set count).
#   4. If the run hits trained mode (with Exact), also: MPRW <= Exact.
#
# If any assertion fails, exits non-zero so the caller knows not to kick
# off the full overnight sweep.
#
# Usage (from project root):
#   bash scripts/smoke_test_edge_counter.sh
#   MODE=trained bash scripts/smoke_test_edge_counter.sh   # slower, also tests Exact
#
# HGB_ACM @ 0.5 chosen because that's where the original "MPRW > Exact"
# anomaly was visible at low w in the transfer/ CSV.

set -euo pipefail

# Force utf-8 stdout — bench_fraction_sweep.py prints unicode glyphs
# (e.g. '∈') in status lines that crash on cp1252 (Windows non-utf8
# locales). Linux is usually fine but we set it everywhere defensively.
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

DATASET="${DATASET:-HGB_ACM}"
FRACTION="${FRACTION:-0.5}"
SEED="${SEED:-42}"
MODE="${MODE:-untrained}"
RESULTS_DIR="results/${DATASET}"
CSV="${RESULTS_DIR}/kgrw_bench_fractions.csv"
BACKUP="${CSV}.pre_smoke_$(date +%s).bak"

echo "=== Smoke test: unified edge counter ==="
echo "Dataset:   ${DATASET}"
echo "Fraction:  ${FRACTION}"
echo "Seed:      ${SEED}"
echo

# Move any existing CSV aside so the smoke run starts clean.
if [[ -f "${CSV}" ]]; then
    echo "[smoke] backing up existing ${CSV} -> ${BACKUP}"
    mv "${CSV}" "${BACKUP}"
fi

# Run the minimal sweep: 2 k's, 2 w's, 1 seed, methods KMV+MPRW only.
# mode=untrained skips GNN training so this completes in seconds.
# (Exact only runs in mode=trained; that's a deliberate design choice.)
echo "[smoke] running bench_fraction_sweep.py (mode=${MODE})..."
python scripts/bench_fraction_sweep.py \
    --dataset "${DATASET}" \
    --mode "${MODE}" \
    --fractions "${FRACTION}" \
    --kmv-k 4 32 \
    --mprw-w 1 32 \
    --seeds 1 \
    --seed-base "${SEED}" \
    --methods KMV MPRW

echo
echo "[smoke] inspecting CSV..."

if [[ ! -f "${CSV}" ]]; then
    echo "[FAIL] CSV not produced at ${CSV}"
    exit 1
fi

# Parse the CSV and assert the cross-method properties.
python - <<PY
import csv, sys

csv_path = "${CSV}"
target_frac = "${FRACTION}"

rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
rows = [r for r in rows if abs(float(r["Fraction"]) - float(target_frac)) < 1e-6]
if not rows:
    print(f"[FAIL] no rows at fraction={target_frac}"); sys.exit(1)

def edges(method, **kw):
    out = []
    for r in rows:
        if r["Method"] != method: continue
        ok = True
        for k, v in kw.items():
            if str(r[k]) != str(v): ok = False; break
        if ok: out.append(int(r["Edge_Count"]))
    return out

exact  = edges("Exact")
kmv4   = edges("KMV", k=4)
kmv32  = edges("KMV", k=32)
mprw1  = edges("MPRW", w_prime=1)
mprw32 = edges("MPRW", w_prime=32)

print(f"  Exact      : {exact}      (only present in trained mode)")
print(f"  KMV  k=4   : {kmv4}")
print(f"  KMV  k=32  : {kmv32}")
print(f"  MPRW w=1   : {mprw1}")
print(f"  MPRW w=32  : {mprw32}")
print()

failures = []

# Always-applicable sanity checks (mode-independent).
if not kmv4 or not kmv32:
    failures.append("KMV rows missing for k=4 or k=32")
elif kmv32[0] < kmv4[0]:
    failures.append(f"KMV non-monotone in k: k=4 -> {kmv4[0]} > k=32 -> {kmv32[0]}")

if not mprw1 or not mprw32:
    failures.append("MPRW rows missing for w=1 or w=32")
elif mprw32[0] < mprw1[0]:
    failures.append(f"MPRW non-monotone in w: w=1 -> {mprw1[0]} > w=32 -> {mprw32[0]}")

# Cross-method bound: MPRW <= KMV at k_max with 10% slack.
# At full saturation both should approach the true neighbourhood; a much
# larger MPRW would suggest the symmetrisation halving didn't take effect.
if kmv32 and mprw32:
    if mprw32[0] > kmv32[0] * 1.10:
        failures.append(f"MPRW w=32 ({mprw32[0]}) > KMV k=32 ({kmv32[0]}) by >10% — "
                        "symmetrisation bug not fixed?")

# Trained-mode-only check: nothing exceeds Exact.
if exact:
    e = exact[0]
    print(f"  (trained mode: Exact={e}, applying tighter checks)")
    if kmv32 and kmv32[0] > e:
        failures.append(f"KMV k=32 ({kmv32[0]}) > Exact ({e}) — sketch can't exceed truth")
    for label, vals in [("MPRW w=1", mprw1), ("MPRW w=32", mprw32)]:
        if vals and vals[0] > e * 1.05:
            failures.append(f"{label} ({vals[0]}) > Exact ({e}) — symmetrisation bug not fixed?")

if failures:
    print("[FAIL] sanity checks:")
    for f in failures: print(f"  - {f}")
    sys.exit(1)
else:
    print("[OK] all sanity checks passed.")
PY

echo
echo "[smoke] done. Safe to kick off the full sweep."
