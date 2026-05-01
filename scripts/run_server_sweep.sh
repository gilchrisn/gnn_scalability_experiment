#!/usr/bin/env bash
#
# Server-side master runner for the journal-extension experiments.
#
# Runs the full pitch pack the prof needs:
#
#   Phase 0   smoke test the unified edge counter
#   Phase 1a  bench_fraction_sweep (HGB × 4, mode=trained)     ← MUST redo (old counter bad)
#   Phase 1b  bench_fraction_sweep (OGB_MAG, mode=untrained)   ← scalability headline
#   Phase 2   sketch-feature NC × 4 HGB × 3 backbones × 3 seeds
#   Phase 3   sketch-sparsifier NC × 4 HGB × 3 seeds
#   Phase 4   simple-HGN baseline (per-task) × 4 HGB × 3 seeds
#   Phase 5   HGSampling baseline × 4 HGB × 3 seeds
#   Phase 6   multi-task SHGN × 4 HGB × 3 seeds   ← unblocked by BFS vectorization
#   Phase 7   sketch-LP × 4 HGB × 3 seeds
#   Phase 8   sketch-similarity × 4 HGB × 1 seed
#   Phase 9   multi-query amortization × 4 HGB × 1 seed
#   Phase 10  compile_master_results.py  (CSV + master tables)
#   Phase 11  plot_session_results.py    (5 paper figures)
#   Phase 12  aggregate_fractions.py     (3-way scatter + saturation)
#
# Per-phase skip flags (set to 1 to skip):
#   SKIP_SMOKE  SKIP_FRACTION  SKIP_SKETCH_NC  SKIP_SPARSIFIER
#   SKIP_SHGN_BASELINE  SKIP_HGSAMPLING  SKIP_MULTITASK
#   SKIP_LP  SKIP_SIMILARITY  SKIP_AMORTIZATION  SKIP_MASTER  SKIP_PLOTS
#
# FORCE_REDO=1 nukes any existing per-task JSON files so phases 2-9
# re-run from scratch. Phase 1 always force-redoes (old CSV is buggy).
#
# Tunables (defaults shown):
#   HGB_DATASETS="HGB_DBLP HGB_ACM HGB_IMDB HNE_PubMed"
#   FRACTION_K_VALUES="2 4 8 16 32 64"
#   FRACTION_W_VALUES="1 2 4 8 16 32 64 128 256 512"
#   FRACTION_SEEDS=3
#   SEED_BASE=42
#   K_VALUE=32
#   NUM_SEEDS=3
#
# Usage (from project root, server-side):
#   bash scripts/run_server_sweep.sh

set -uo pipefail
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# ── Defaults ────────────────────────────────────────────────────────────
HGB_DATASETS="${HGB_DATASETS:-HGB_DBLP HGB_ACM HGB_IMDB HNE_PubMed}"
FRACTION_K_VALUES="${FRACTION_K_VALUES:-2 4 8 16 32 64}"
FRACTION_W_VALUES="${FRACTION_W_VALUES:-1 2 4 8 16 32 64 128 256 512}"
FRACTION_SEEDS="${FRACTION_SEEDS:-3}"
SEED_BASE="${SEED_BASE:-42}"
K_VALUE="${K_VALUE:-32}"
NUM_SEEDS="${NUM_SEEDS:-3}"
FORCE_REDO="${FORCE_REDO:-0}"

# Per-seed list (shell loop) — used by per-task scripts that take --seed.
SEEDS=()
for ((i=0; i<NUM_SEEDS; i++)); do SEEDS+=( $((SEED_BASE + i)) ); done

# ── Logging ─────────────────────────────────────────────────────────────
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/server_run_${TS}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${MASTER_LOG}"; }
section() { log ""; log "==========================================================";
            log "$*"; log "=========================================================="; }

# ── Helper: skip-if-JSON-exists guard for per-task scripts ─────────────
# Usage: should_skip <json_path>
should_skip() {
    local p="$1"
    if [[ "${FORCE_REDO}" == "1" ]]; then
        rm -f "${p}"
        return 1
    fi
    [[ -f "${p}" ]]
}

run_or_skip() {
    # $1 = label, $2 = json_path, rest = command
    local label="$1"; local json_path="$2"; shift 2
    if should_skip "${json_path}"; then
        log "  ${label}: SKIP (exists at ${json_path})"
        return 0
    fi
    log "  ${label}: running..."
    local lf="${LOG_DIR}/$(basename ${json_path%.json}).${TS}.log"
    if "$@" 2>&1 | tee "${lf}" | tail -5 >> "${MASTER_LOG}"; then
        log "  ${label}: OK"
    else
        log "  ${label}: FAILED (log: ${lf})"
    fi
}

section "Master log: ${MASTER_LOG}"
log "HGB datasets:   ${HGB_DATASETS}"
log "Seeds:          ${SEEDS[*]}  (NUM_SEEDS=${NUM_SEEDS}, SEED_BASE=${SEED_BASE})"
log "k value:        ${K_VALUE}"
log "FORCE_REDO:     ${FORCE_REDO}"

# ════════════════════════════════════════════════════════════════════════
# PHASE 0: smoke test
# ════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
    section "PHASE 0 — smoke test edge counter"
    if bash scripts/smoke_test_edge_counter.sh 2>&1 | tee -a "${MASTER_LOG}"; then
        log "[smoke] PASS"
    else
        log "[smoke] FAIL — aborting"
        exit 1
    fi
else
    log "[skip] phase 0 (smoke)"
fi

# ════════════════════════════════════════════════════════════════════════
# PHASE 1: fraction sweep — MUST redo because old CSV had wrong edge counts
# ════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_FRACTION:-0}" != "1" ]]; then
    section "PHASE 1 — bench_fraction_sweep (force-redo, old counter was bad)"

    # Backup any existing CSVs once at phase start so we don't lose history.
    for ds in ${HGB_DATASETS} OGB_MAG; do
        old_csv="results/${ds}/kgrw_bench_fractions.csv"
        if [[ -f "${old_csv}" ]]; then
            mv "${old_csv}" "${old_csv}.pre_${TS}.bak"
            log "  backed up ${old_csv} → ${old_csv}.pre_${TS}.bak"
        fi
    done

    # Phase 1a: HGB datasets in TRAINED mode (matches old transfer/ data).
    log ""
    log "─── Phase 1a: HGB fraction sweep (mode=trained, ${FRACTION_SEEDS} seeds) ───"
    for ds in ${HGB_DATASETS}; do
        log ""
        log "  --- ${ds} ---"
        ds_log="${LOG_DIR}/${ds}/fraction_sweep_${TS}.log"
        mkdir -p "${LOG_DIR}/${ds}"
        if python scripts/bench_fraction_sweep.py \
            --dataset "${ds}" \
            --mode trained \
            --kmv-k ${FRACTION_K_VALUES} \
            --mprw-w ${FRACTION_W_VALUES} \
            --seeds "${FRACTION_SEEDS}" \
            --seed-base "${SEED_BASE}" \
            2>&1 | tee "${ds_log}" | tail -20 >> "${MASTER_LOG}"; then
            log "  ${ds}: OK   (full log: ${ds_log})"
        else
            log "  ${ds}: FAILED   (full log: ${ds_log})"
        fi
    done

    # Phase 1b: OGB-MAG in UNTRAINED mode (scalability headline).
    log ""
    log "─── Phase 1b: OGB-MAG fraction sweep (mode=untrained, ${FRACTION_SEEDS} seeds) ───"
    log ""
    log "  --- OGB_MAG ---"
    ds_log="${LOG_DIR}/OGB_MAG/fraction_sweep_${TS}.log"
    mkdir -p "${LOG_DIR}/OGB_MAG"
    if python scripts/bench_fraction_sweep.py \
        --dataset OGB_MAG \
        --mode untrained \
        --kmv-k ${FRACTION_K_VALUES} \
        --mprw-w ${FRACTION_W_VALUES} \
        --seeds "${FRACTION_SEEDS}" \
        --seed-base "${SEED_BASE}" \
        2>&1 | tee "${ds_log}" | tail -20 >> "${MASTER_LOG}"; then
        log "  OGB_MAG: OK   (full log: ${ds_log})"
    else
        log "  OGB_MAG: FAILED   (full log: ${ds_log})"
    fi
else
    log "[skip] phase 1 (fraction sweep)"
fi

# ════════════════════════════════════════════════════════════════════════
# PHASE 2: sketch-feature NC × 3 backbones × 4 HGB × 3 seeds
# ════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_SKETCH_NC:-0}" != "1" ]]; then
    section "PHASE 2 — sketch-feature NC (3 backbones × ${NUM_SEEDS} seeds)"
    for ds in ${HGB_DATASETS}; do
        for backbone in mlp han_sketch_edges; do
            # han_real_edges is OOM-prone on dense graphs; skip by default.
            for seed in "${SEEDS[@]}"; do
                if [[ "${backbone}" == "mlp" ]]; then
                    json="results/${ds}/sketch_feature_pilot_k${K_VALUE}_mlp_seed${seed}.json"
                else
                    json="results/${ds}/sketch_feature_pilot_k${K_VALUE}_seed${seed}.json"
                fi
                run_or_skip "${ds} backbone=${backbone} seed=${seed}" "${json}" \
                    python scripts/exp_sketch_feature_train.py "${ds}" \
                        --k "${K_VALUE}" --seed "${seed}" --backbone "${backbone}"
            done
        done
    done
else
    log "[skip] phase 2 (sketch-feature NC)"
fi

# ════════════════════════════════════════════════════════════════════════
# PHASE 3: sketch-sparsifier NC × 4 HGB × 3 seeds
# ════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_SPARSIFIER:-0}" != "1" ]]; then
    section "PHASE 3 — sketch-sparsifier NC (${NUM_SEEDS} seeds)"
    for ds in ${HGB_DATASETS}; do
        for seed in "${SEEDS[@]}"; do
            json="results/${ds}/sketch_sparsifier_pilot_k${K_VALUE}_seed${seed}.json"
            run_or_skip "${ds} seed=${seed}" "${json}" \
                python scripts/exp_sketch_sparsifier_train.py "${ds}" \
                    --k "${K_VALUE}" --seed "${seed}"
        done
    done
else
    log "[skip] phase 3 (sparsifier)"
fi

# ════════════════════════════════════════════════════════════════════════
# PHASE 4: simple-HGN baseline (per-task) × 4 HGB
# (script handles --num-seeds internally)
# ════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_SHGN_BASELINE:-0}" != "1" ]]; then
    section "PHASE 4 — simple-HGN baseline per-task (${NUM_SEEDS} seeds, internal loop)"
    for ds in ${HGB_DATASETS}; do
        # Skip if every expected per-seed JSON exists.
        all_done=1
        for seed in "${SEEDS[@]}"; do
            [[ -f "results/${ds}/simple_hgn_baseline_seed${seed}.json" ]] || { all_done=0; break; }
        done
        if [[ "${all_done}" == "1" && "${FORCE_REDO}" != "1" ]]; then
            log "  ${ds}: SKIP (all seeds present)"
            continue
        fi
        if [[ "${FORCE_REDO}" == "1" ]]; then
            for seed in "${SEEDS[@]}"; do rm -f "results/${ds}/simple_hgn_baseline_seed${seed}.json"; done
        fi
        log "  ${ds}: running..."
        ds_log="${LOG_DIR}/${ds}/shgn_baseline_${TS}.log"
        mkdir -p "${LOG_DIR}/${ds}"
        if python scripts/exp_simple_hgn_baseline.py "${ds}" \
            --num-seeds "${NUM_SEEDS}" --seed-base "${SEED_BASE}" \
            2>&1 | tee "${ds_log}" | tail -10 >> "${MASTER_LOG}"; then
            log "  ${ds}: OK"
        else
            log "  ${ds}: FAILED (log: ${ds_log})"
        fi
    done
else
    log "[skip] phase 4 (simple-HGN baseline)"
fi

# ════════════════════════════════════════════════════════════════════════
# PHASE 5: HGSampling baseline × 4 HGB
# ════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_HGSAMPLING:-0}" != "1" ]]; then
    section "PHASE 5 — HGSampling baseline (${NUM_SEEDS} seeds, internal loop)"
    for ds in ${HGB_DATASETS}; do
        all_done=1
        for seed in "${SEEDS[@]}"; do
            [[ -f "results/${ds}/hgsampling_baseline_seed${seed}.json" ]] || { all_done=0; break; }
        done
        if [[ "${all_done}" == "1" && "${FORCE_REDO}" != "1" ]]; then
            log "  ${ds}: SKIP (all seeds present)"
            continue
        fi
        if [[ "${FORCE_REDO}" == "1" ]]; then
            for seed in "${SEEDS[@]}"; do rm -f "results/${ds}/hgsampling_baseline_seed${seed}.json"; done
        fi
        log "  ${ds}: running..."
        ds_log="${LOG_DIR}/${ds}/hgsampling_${TS}.log"
        mkdir -p "${LOG_DIR}/${ds}"
        if python scripts/exp_hgsampling_baseline.py "${ds}" \
            --num-seeds "${NUM_SEEDS}" --seed-base "${SEED_BASE}" \
            2>&1 | tee "${ds_log}" | tail -10 >> "${MASTER_LOG}"; then
            log "  ${ds}: OK"
        else
            log "  ${ds}: FAILED (log: ${ds_log})"
        fi
    done
else
    log "[skip] phase 5 (HGSampling)"
fi

# ════════════════════════════════════════════════════════════════════════
# PHASE 6: multi-task SHGN × 4 HGB
# ════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_MULTITASK:-0}" != "1" ]]; then
    section "PHASE 6 — multi-task SHGN (${NUM_SEEDS} seeds, internal loop)"
    for ds in ${HGB_DATASETS}; do
        partition="results/${ds}/partition.json"
        if [[ ! -f "${partition}" ]]; then
            log "  ${ds}: SKIP — partition.json missing at ${partition}"
            continue
        fi
        all_done=1
        for seed in "${SEEDS[@]}"; do
            [[ -f "results/${ds}/simple_hgn_multitask_seed${seed}.json" ]] || { all_done=0; break; }
        done
        if [[ "${all_done}" == "1" && "${FORCE_REDO}" != "1" ]]; then
            log "  ${ds}: SKIP (all seeds present)"
            continue
        fi
        if [[ "${FORCE_REDO}" == "1" ]]; then
            for seed in "${SEEDS[@]}"; do rm -f "results/${ds}/simple_hgn_multitask_seed${seed}.json"; done
        fi
        log "  ${ds}: running..."
        ds_log="${LOG_DIR}/${ds}/multitask_${TS}.log"
        mkdir -p "${LOG_DIR}/${ds}"
        if python scripts/exp_simple_hgn_multitask.py "${ds}" \
            --partition-json "${partition}" \
            --num-seeds "${NUM_SEEDS}" --seed-base "${SEED_BASE}" \
            2>&1 | tee "${ds_log}" | tail -10 >> "${MASTER_LOG}"; then
            log "  ${ds}: OK"
        else
            log "  ${ds}: FAILED (log: ${ds_log})"
        fi
    done
else
    log "[skip] phase 6 (multi-task SHGN)"
fi

# ════════════════════════════════════════════════════════════════════════
# PHASE 7: sketch-LP × 4 HGB × 3 seeds
# ════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_LP:-0}" != "1" ]]; then
    section "PHASE 7 — sketch-LP (${NUM_SEEDS} seeds)"
    for ds in ${HGB_DATASETS}; do
        partition="results/${ds}/partition.json"
        for seed in "${SEEDS[@]}"; do
            json="results/${ds}/sketch_lp_pilot_k${K_VALUE}_seed${seed}.json"
            if [[ ! -f "${partition}" ]]; then
                log "  ${ds} seed=${seed}: SKIP — partition.json missing"
                continue
            fi
            run_or_skip "${ds} seed=${seed}" "${json}" \
                python scripts/exp_sketch_lp_train.py "${ds}" \
                    --partition-json "${partition}" \
                    --k "${K_VALUE}" --seed "${seed}"
        done
    done
else
    log "[skip] phase 7 (sketch-LP)"
fi

# ════════════════════════════════════════════════════════════════════════
# PHASE 8: sketch-similarity × 4 HGB × 1 seed (Jaccard fidelity is
# deterministic per seed; one seed is the standard report)
# ════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_SIMILARITY:-0}" != "1" ]]; then
    section "PHASE 8 — sketch-similarity Jaccard fidelity (1 seed)"
    for ds in ${HGB_DATASETS}; do
        json="results/${ds}/sketch_similarity_pilot_k${K_VALUE}_seed${SEED_BASE}.json"
        run_or_skip "${ds} seed=${SEED_BASE}" "${json}" \
            python scripts/exp_sketch_similarity.py "${ds}" \
                --k "${K_VALUE}" --seed "${SEED_BASE}"
    done
else
    log "[skip] phase 8 (similarity)"
fi

# ════════════════════════════════════════════════════════════════════════
# PHASE 9: multi-query amortization × 4 HGB × 1 seed
# ════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_AMORTIZATION:-0}" != "1" ]]; then
    section "PHASE 9 — multi-query amortization (1 seed)"
    for ds in ${HGB_DATASETS}; do
        json="results/${ds}/multi_query_amortization_k${K_VALUE}_seed${SEED_BASE}.json"
        run_or_skip "${ds} seed=${SEED_BASE}" "${json}" \
            python scripts/exp_multi_query_amortization.py "${ds}" \
                --k "${K_VALUE}" --seed "${SEED_BASE}"
    done
else
    log "[skip] phase 9 (amortization)"
fi

# ════════════════════════════════════════════════════════════════════════
# PHASE 10: compile master tables
# ════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_MASTER:-0}" != "1" ]]; then
    section "PHASE 10 — compile master tables"
    if python scripts/compile_master_results.py 2>&1 | tee -a "${MASTER_LOG}"; then
        log "[master] OK"
    else
        log "[master] FAILED"
    fi
else
    log "[skip] phase 10 (master tables)"
fi

# ════════════════════════════════════════════════════════════════════════
# PHASE 11: paper figures (5 PNG/PDF)
# ════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_PLOTS:-0}" != "1" ]]; then
    section "PHASE 11 — paper figures (plot_session_results.py)"
    if python scripts/plot_session_results.py 2>&1 | tee -a "${MASTER_LOG}"; then
        log "[plots] OK"
    else
        log "[plots] FAILED"
    fi

    section "PHASE 12 — fraction-sweep aggregate plots"
    if python scripts/aggregate_fractions.py \
        --datasets ${HGB_DATASETS} OGB_MAG \
        --root-dir results --out-dir figures/server_run \
        2>&1 | tee -a "${MASTER_LOG}"; then
        log "[fraction plots] OK"
    else
        log "[fraction plots] FAILED"
    fi
else
    log "[skip] phases 11-12 (plots)"
fi

section "DONE"
log "Master log:                ${MASTER_LOG}"
log "Per-task JSONs:            results/<DATASET>/*.json"
log "Fraction sweep CSVs:       results/<DATASET>/kgrw_bench_fractions.csv"
log "Master tables:             results/master_table_*.md"
log "Paper figures:             figures/sketch_session/*.png"
log "Fraction-sweep plots:      figures/server_run/*.pdf"
