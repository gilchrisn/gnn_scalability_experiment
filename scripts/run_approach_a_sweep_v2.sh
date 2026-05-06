#!/usr/bin/env bash
# run_approach_a_sweep_v2.sh
#
# v2 main sweep orchestrator for the Approach A fidelity experiment.
# Spec: final_report/research_notes/SPEC_approach_a_2026_05_07.md
#
# Reads results/convergence_matrix.csv and runs only cells whose verdict is
# CONVERGED. For each (dataset, arch, mp), iterates over seeds and k values
# and invokes scripts/exp3_inference.py once per (seed, k). Resume-safe: any
# (dataset, arch, mp_safe, seed, k) JSON that already exists in the v2 output
# directory is skipped.
#
# Usage:
#   bash scripts/run_approach_a_sweep_v2.sh
#   bash scripts/run_approach_a_sweep_v2.sh --seeds 42 43 44 45 46
#   bash scripts/run_approach_a_sweep_v2.sh --seeds-headline   # 10 seeds for the four headline cells
#   bash scripts/run_approach_a_sweep_v2.sh --include-sanity   # also run sanity controls after the main sweep
#
# Expected wall-time: ~30 h on the yudong server for the default 5-seed sweep.

set -euo pipefail
cd "$(dirname "$0")/.."

# ---- defaults --------------------------------------------------------------

DEFAULT_SEEDS=(42 43 44 45 46)
HEADLINE_SEEDS=(42 43 44 45 46 47 48 49 50 51)
HEADLINE_CELLS=(
  "HGB_DBLP|SAGE|APA"
  "HGB_ACM|SAGE|PAP"
  "HGB_ACM|GCN|PAP"
  "HGB_ACM|GAT|PAP"
)

K_VALUES=(8 16 32 64 128)
DEPTH=2
CONV_CSV="results/convergence_matrix.csv"
OUT_ROOT="results/approach_a_2026_05_07"
LOG_DIR="results"
LOG="$LOG_DIR/approach_a_sweep_v2_$(date +%Y%m%d_%H%M%S).log"

INCLUDE_SANITY=0
USE_HEADLINE_SEEDS=0
SEEDS=("${DEFAULT_SEEDS[@]}")

# ---- arg parse -------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      shift
      SEEDS=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        SEEDS+=("$1")
        shift
      done
      ;;
    --seeds-headline)
      USE_HEADLINE_SEEDS=1
      shift
      ;;
    --include-sanity)
      INCLUDE_SANITY=1
      shift
      ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$LOG_DIR"
echo "[$(date)] starting Approach A v2 sweep" | tee -a "$LOG"
echo "[$(date)] seeds=${SEEDS[*]} headline_seeds=${HEADLINE_SEEDS[*]}" | tee -a "$LOG"
echo "[$(date)] k_values=${K_VALUES[*]} depth=$DEPTH" | tee -a "$LOG"
echo "[$(date)] convergence_csv=$CONV_CSV out_root=$OUT_ROOT" | tee -a "$LOG"
echo "[$(date)] include_sanity=$INCLUDE_SANITY use_headline_seeds=$USE_HEADLINE_SEEDS" | tee -a "$LOG"

# ---- mp short -> mp full mapping -------------------------------------------

declare -A MP_FULL
MP_FULL["HGB_DBLP/APA"]="author_to_paper,paper_to_author"
MP_FULL["HGB_DBLP/APVPA"]="author_to_paper,paper_to_venue,venue_to_paper,paper_to_author"
MP_FULL["HGB_ACM/PAP"]="paper_to_author,author_to_paper"
MP_FULL["HGB_ACM/PTP"]="paper_to_term,term_to_paper"
MP_FULL["HGB_IMDB/MAM"]="movie_to_actor,actor_to_movie"
MP_FULL["HGB_IMDB/MKM"]="movie_to_keyword,keyword_to_movie"
MP_FULL["HNE_PubMed/DGD"]="disease_to_gene,gene_to_disease"
MP_FULL["HNE_PubMed/DCD"]="disease_to_chemical,chemical_to_disease"

mp_safe() {
  # convert "a,b,c" -> "a_b_c"
  echo "$1" | tr ',' '_'
}

is_headline_cell() {
  local key="$1"
  for h in "${HEADLINE_CELLS[@]}"; do
    if [[ "$h" == "$key" ]]; then
      return 0
    fi
  done
  return 1
}

# ---- collect converged cells from convergence_matrix.csv -------------------

if [[ ! -f "$CONV_CSV" ]]; then
  echo "[$(date)] FATAL: $CONV_CSV not found" | tee -a "$LOG" >&2
  exit 3
fi

# CSV columns: dataset,arch,mp,verdict,...
CONVERGED_ROWS=$(awk -F, 'NR > 1 && $4 == "CONVERGED" { print $1 "|" $2 "|" $3 }' "$CONV_CSV")
n_cells=$(echo "$CONVERGED_ROWS" | grep -c '|' || true)
echo "[$(date)] found $n_cells converged cells in $CONV_CSV" | tee -a "$LOG"

# ---- compute total runs ----------------------------------------------------

n_runs_total=0
while IFS='|' read -r ds arch mp_short; do
  [[ -z "$ds" ]] && continue
  cell_key="$ds/$mp_short"
  if [[ -z "${MP_FULL[$cell_key]:-}" ]]; then
    continue
  fi
  cell_id="$ds|$arch|$mp_short"
  if [[ "$USE_HEADLINE_SEEDS" -eq 1 ]] && is_headline_cell "$cell_id"; then
    seeds_for_cell=("${HEADLINE_SEEDS[@]}")
  else
    seeds_for_cell=("${SEEDS[@]}")
  fi
  n_runs_total=$(( n_runs_total + ${#seeds_for_cell[@]} * ${#K_VALUES[@]} ))
done <<< "$CONVERGED_ROWS"
echo "[$(date)] total planned runs (main sweep): $n_runs_total" | tee -a "$LOG"

# ---- main sweep ------------------------------------------------------------

run_idx=0
n_ok=0
n_skip=0
n_fail=0

while IFS='|' read -r ds arch mp_short; do
  [[ -z "$ds" ]] && continue
  cell_key="$ds/$mp_short"
  mp_full="${MP_FULL[$cell_key]:-}"
  if [[ -z "$mp_full" ]]; then
    echo "[$(date)] WARN: no MP_FULL mapping for $cell_key, skipping" | tee -a "$LOG"
    continue
  fi
  mps=$(mp_safe "$mp_full")
  cell_id="$ds|$arch|$mp_short"

  if [[ "$USE_HEADLINE_SEEDS" -eq 1 ]] && is_headline_cell "$cell_id"; then
    seeds_for_cell=("${HEADLINE_SEEDS[@]}")
  else
    seeds_for_cell=("${SEEDS[@]}")
  fi

  for seed in "${seeds_for_cell[@]}"; do
    for k in "${K_VALUES[@]}"; do
      run_idx=$(( run_idx + 1 ))
      out_json="$OUT_ROOT/$ds/$arch/${mps}_seed${seed}_k${k}.json"
      tag="$ds/$arch/$mp_short seed=$seed k=$k"

      if [[ -f "$out_json" ]]; then
        n_skip=$(( n_skip + 1 ))
        echo "[$run_idx/$n_runs_total] $tag  SKIP (exists)" | tee -a "$LOG"
        continue
      fi

      mkdir -p "$(dirname "$out_json")"

      echo "[$run_idx/$n_runs_total] $tag  RUN" | tee -a "$LOG"
      set +e
      python scripts/exp3_inference.py "$ds" \
        --metapath "$mp_full" \
        --depth "$DEPTH" \
        --k-values "$k" \
        --partition-json "results/$ds/partition.json" \
        --weights-dir "results/$ds/weights" \
        --output "results/$ds/master_results_v2.csv" \
        --arch "$arch" \
        --skip-mprw \
        --run-id "$seed" \
        --hash-seed "$seed" >>"$LOG" 2>&1
      rc=$?
      set -e

      if [[ $rc -ne 0 ]]; then
        n_fail=$(( n_fail + 1 ))
        echo "[$run_idx/$n_runs_total] $tag  FAIL rc=$rc" | tee -a "$LOG" >&2
        continue
      fi

      if [[ -f "$out_json" ]]; then
        n_ok=$(( n_ok + 1 ))
        echo "[$run_idx/$n_runs_total] $tag  OK" | tee -a "$LOG"
      else
        n_fail=$(( n_fail + 1 ))
        echo "[$run_idx/$n_runs_total] $tag  WARN no JSON at $out_json" | tee -a "$LOG" >&2
      fi
    done
  done
done <<< "$CONVERGED_ROWS"

echo "" | tee -a "$LOG"
echo "[$(date)] main sweep done: ok=$n_ok skip=$n_skip fail=$n_fail total=$run_idx" | tee -a "$LOG"

# ---- sanity controls (optional) --------------------------------------------

if [[ "$INCLUDE_SANITY" -eq 1 ]]; then
  echo "" | tee -a "$LOG"
  echo "[$(date)] starting sanity controls (DBLP-APA-SAGE)" | tee -a "$LOG"

  SANITY_DS="HGB_DBLP"
  SANITY_TGT="author"
  SANITY_MP="author_to_paper,paper_to_author"
  SANITY_ARCH="SAGE"
  SANITY_SEEDS=(42 43 44)
  SANITY_MODES=(random_theta edge_perturb density_matched_random layer_permutation)

  for mode in "${SANITY_MODES[@]}"; do
    echo "[$(date)] sanity mode=$mode" | tee -a "$LOG"
    set +e
    python scripts/exp_sanity_controls.py \
      --mode "$mode" \
      --dataset "$SANITY_DS" \
      --metapath "$SANITY_MP" \
      --target-type "$SANITY_TGT" \
      --arch "$SANITY_ARCH" \
      --depth "$DEPTH" \
      --k-values "${K_VALUES[@]}" \
      --seeds "${SANITY_SEEDS[@]}" \
      --partition-json "results/$SANITY_DS/partition.json" \
      --weights-dir "results/$SANITY_DS/weights" >>"$LOG" 2>&1
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      echo "[$(date)] sanity mode=$mode FAIL rc=$rc" | tee -a "$LOG" >&2
    else
      echo "[$(date)] sanity mode=$mode OK" | tee -a "$LOG"
    fi
  done
fi

echo "[$(date)] DONE log=$LOG" | tee -a "$LOG"
