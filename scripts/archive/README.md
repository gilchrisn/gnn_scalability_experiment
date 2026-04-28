# scripts/archive/

Code from abandoned research branches. **Do not run, do not import from active scripts.** Kept for audit trail — the experiments happened, findings are preserved in `final_report/` and `memory/`, and the scripts themselves document methodology that may be cited.

If you need any of these in the active set again, copy back and update `CURRENT_STATE.md` to reflect the new scope.

---

## Path 1 status (2026-04-28)

The locked paper scope is **VLDB-J extension as multi-query amortization framework** (see `final_report/research_notes/CURRENT_STATE.md`). Anything in this archive is *outside Path 1 scope*. The reasons each branch was abandoned are documented below.

---

## KGRW chain (abandoned per `memory/project_kgrw_findings.md`)

KGRW (Knowledge-Graph Random Walk) was a hybrid sketch+walk variant tested as a possible improvement over MPRW. All evidence falsified the hypothesis: dead-walks-free advantage was not real, round-robin ceiling capped quality, Phase 2 saturation was not avoided. Abandoned 2026-04-19 region.

- `bench_kgrw.py` — main KGRW benchmark across seeds.
- `bench_metapath_sweep.py` — orchestrator that calls `bench_kgrw.py` per metapath.
- `eval_kgrw.py` — KGRW evaluation utility.
- `aggregate_kgrw_bench.py` — KGRW CSV aggregation.
- `gen_kgrw_plots.py` — KGRW figures.
- `aggregate_3way_l2.py` — KMV / KGRW / MPRW combined L=2 analysis.
- `tail_fraction.py` — measures |T|/|V_L| from exact metapath; used only by `gen_kgrw_plots.py`.
- `method_saturation.py` — KMV / KGRW / MPRW saturation analysis (KGRW component dead).
- `analysis_is_bias.py` — IS-bias diagnostic comparing MPRW vs KGRW.
- `kmv_saturation.py` — reads `kgrw_bench.csv` for KMV saturation curves; methodology fine but reads dead CSV. Path 1 will redo from `master_results.csv`.

## IS-bias / spectral / DBLP-APTPA work (pre-Path-1, abandoned)

The "KMV beats MPRW because MPRW has IS bias" rescue narrative. Tested empirically; falsified at matched density on standard meta-paths. Survived only as a narrow regime finding (DBLP-APTPA L=4 starved budget) — see `memory/project_dblp_aptpa_is_regime.md`. Out of Path 1 scope.

- `exp6_table5_stats.py` — densest-metapath-only Table 5 variant. Superseded by `../table5_stats.py` (all-metapaths).
- `exp6_is_vs_us.py` — "MPRW ≈ IS, KMV ≈ uniform" hypothesis test on HNE_PubMed.
- `exp7_spectral.py` — spectral comparison Exact vs KMV vs MPRW. Falsified per `memory/project_hgnn_extension_abandoned.md`.
- `exp8_importance_sampling.py` — direct empirical test of MPRW = IS.
- `exp9_plot_dblp_aptpa.py` — DBLP APTPA narrow-regime plot.
- `exp10_is_mechanism_dblp.py` — direct measurement of IS bias on DBLP APTPA.

## Mechanistic invariance investigation (pre-Path-1, abandoned)

The "investigate what topology K_L[v] preserves" line of work. Theorem D came out of this and is preserved in `final_report/research_notes/12_separation_theorem_attempt.md` + `14_theorem_D_multilayer_extension.md`. The empirical scripts are out of Path 1 scope; if a separate mechanistic paper happens later, these are the starting point.

- `exp11_kmv_invariances.py` — I1-I4 KMV invariance measurements.
- `exp12_mprw_bias.py` — H1 MPRW homophily bias measurement.
- `exp13_kmv_properties_plots.py` — visualization for exp11/12.
- `exp14_metapath_profile.py` — per-metapath structural profiling.
- `exp15_h1_crossdataset.py` — H1 across HGB datasets.
- `exp16_coupon_verification.py` — coupon-collector verification of KMV behavior.
- `exp17_regime_characterization.py` — regime classification (reads kgrw_bench.csv).
- `exp18_rigor_check.py` — robustness check for invariance claims.

## Old orchestrator (pre-refactor)

- `run_extension_experiments.py` — original multi-metapath extension orchestrator. Predates exp1/exp2/exp3/exp4 + `run_full_pipeline.py`. Trains SAGE per-metapath instead of using the inductive transfer protocol. Replaced by the current pipeline.

## Pre-existing (already in archive)

These were here before this cleanup — left untouched.

- `*.dat`, `*.limit` — staged graph data from prior runs.
- `rerun_mprw.{ps1,sh}`, `run_mprw_*.{ps1,sh}`, `run_seeds.{ps1,sh}`, `run_exp1_exp2.sh`, `run_exp5.ps1`, `run_large_*.sh`, `setup_server.sh` — older shell orchestration.

---

## How to revive a script

1. Copy back to `scripts/` (don't move — keep audit trail).
2. Update its top-of-file docstring with the date and reason for revival.
3. Update `CURRENT_STATE.md` if the revival reflects a scope change.
4. Verify dependencies (some archived scripts import other archived scripts).
