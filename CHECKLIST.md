# CHECKLIST (independent audit, fresh subagent, then fixes)
| # | Requirement | Result | Evidence |
|---|---|---|---|
| 1 | Four models, binary + 8-class | PASS | results/baseline_config.json, metrics_{dt,rf,xgb,mlp}_{binary,multi}.json |
| 2 | Same split for all | PASS | data_processed/split.npz, src/data.py load() |
| 3 | Macro F1, acc, precision, recall, FPR, CM | PASS | metrics_*.json, cm_*.csv/png, pred_*.npz |
| 4 | Identical-condition resource measurement | PASS (rf_multi baseline OOM at 1 GB, recorded) | bench.py, results/benchmark_docker.csv (official), benchmark_native.csv |
| 5 | Planned reductions | PASS | tree_reduction_grid.csv, tree_selected_variants.csv, metrics_mlp_*.json |
| 6 | Original vs reduced, plots, 3 seeds | PASS | reduced_vs_original_*.csv, figures/pareto_*.png, seed_stability.csv |
| 7 | Leakage | PASS | verify.py (+ --leak), test_access.log (80 entries at the end of phase 1, 176 after phase 2; no duplicates) |
| 8 | Reproducibility | FIXED after audit | requirements.txt pinned; q8dyn code path added in src/reduce_mlp.py; DECISIONS.md renumbered |
| 9 | Report claims supported | PASS | REPORT.md vs results/ files |

Audit findings fixed: unpinned requirements; q8dyn models had no producing code; DECISIONS.md duplicate numbers/stale Docker entry; REPORT row for 8-class distilled+int8 now shows the validation-chosen dynamic-range variant; precision/recall averaging convention documented. Not changed (disclosed in REPORT.md): rf_multi baseline missing from Docker table (OOM), CPU util non-discriminating, runtime-dominated RAM, benchmark input = first 20,000 test rows (features only, no labels used).

## Phase 2 (optimization) checklist
| # | Requirement | Result | Evidence |
|---|---|---|---|
| 10 | One shared encoding, fitness on train/val subsamples only | PASS | optimization/core.py (verify.py: no test reference in code) |
| 11 | GA, PSO, ACO(R), random; equal pop/seed/budget | PASS (binary RF-PSO stopped at 689/800, cache saturation, disclosed) | results/optimization/runs/, budgets.json |
| 12 | Winners retrained, registered, scored once on test (+seeds 43/44) | PASS | results/optimization/final/, test_access.log (176 lines) |
| 13 | Docker 1 CPU / 1 GB benchmark | PASS (rf_multi aco/ga/pso OOM recorded) | results/optimization/benchmark_docker.csv |
| 14 | Notebook with results-at-a-glance table, deltas, bar chart | PASS | iot_ids_optimization.ipynb |
| 15 | No pre-existing result changed | PASS | hash check in verify.py |
| 16 | Honest reporting | PASS | REPORT.md section 9: no gain for XGBoost / 8-class MLP |
Independent audit of this phase: see the audit note at the end of this file.
