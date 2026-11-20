# Lightweight ML for IoT intrusion detection: results report

All numbers are read from files in `results/`; the notebook `iot_ids.ipynb` shows the same tables.

## 1. Setup
* Data: CICIoT2023 `MERGED_CSV` (63 files, 39 features + `Label`). Stratified sample of ~1.06 M rows (`results/sampling_counts.csv`; rule in the notebook: labels with <30,000 rows kept whole, the others proportional to size with a 10,000 floor). After cleaning (`results/cleaning_log.md`): 758,914 rows, 8 classes (`results/clean_class_counts.csv`).
* One 70/15/15 split stratified on the 8-class label, seed 42, shared by all models (`data_processed/split.npz`, `results/split_note.txt`). Scaler fitted on train only. Tuning on validation only; every frozen model scored once on test (`test_access.log`).
* Models: Decision Tree, Random Forest, XGBoost, MLP (Keras), binary and 8-class (`results/baseline_config.json`). Metrics: `results/metrics_*.json`; predictions `results/pred_*.npz`; confusion matrices `results/cm_*.csv` and `figures/cm_*.png`.
* Official benchmark: **Docker, one container per model, `--cpus=1 --memory=1g` (no swap)**, models and the 20,000-row input mounted read-only (`results/benchmark_docker.csv`). Secondary: **native, not Docker-limited**, pinned to one logical CPU (`results/benchmark_native.csv`). Both: 1 thread everywhere, 200 warm-ups, 10,000 batch-1 predictions, >=25 batch-1024 repeats (median).
* Host: Windows 11, Intel i5-12500H (12 cores / 16 threads), 15.6 GB RAM, Docker Desktop VM (16 CPUs, 7.6 GiB); Python 3.11.9, TensorFlow 2.21, scikit-learn 1.9.1, XGBoost 3.2.0.

## 2. Baselines (`results/baseline_comparison_docker.csv`)
| Model | bin macro F1 | 8-class macro F1 | size KB (bin / multi) | batch-1 median ms (bin) | b1024 pred/s (bin) | peak RSS MB (bin) |
|---|---|---|---|---|---|---|
| Decision Tree | 0.785 | 0.708 | 719 / 2,764 | 0.161 | 4.46 M | 131 |
| Random Forest | 0.802 | 0.740 | 139,890 / 1,431,589 (**OOM at 1 GB**) | 4.46 | 76 k | 407 |
| XGBoost | **0.819** | **0.755** | 5,490 / 43,514 | 0.542 | 157 k | 206 |
| MLP (Keras) | 0.780 | 0.710 | 184 / 189 | 1.86 | 403 k | 566 |

Precision and recall are macro-averaged over the classes (two classes in the binary task), while binary FPR uses attack as the positive class, so recall and FPR are different conventions. Other metrics (accuracy, precision, recall, FPR, per-class F1) are in `results/metrics_*.json` and the notebook. Binary FPR (benign flagged as attack) is 0.40-0.51 for all models; benign is only ~2.8% of the data and overlaps attack traffic in these features.

## 3. Reductions (`results/reduced_vs_original_docker.csv`, `results/tree_reduction_grid.csv`, `results/tree_selected_variants.csv`)
Moderate = smallest variant within 1 pt of the baseline *validation* macro F1; aggressive = within 3 pt. Ratios are against the original (Docker).

| Variant | d macro F1 (test) | size | latency | throughput |
|---|---|---|---|---|
| dt_binary_mod (= agg) | -1.0 pt | 0.005x | 0.59x | 1.19x |
| rf_binary_mod / agg | -1.1 / -1.4 pt | 0.023x / 0.009x | 0.28x / 0.14x | 6.2x / 12.6x |
| xgb_binary_mod / agg | -0.7 / -3.0 pt | 0.14x / 0.036x | 0.27x / 0.20x | 4.2x / 12.2x |
| dt_multi_mod (= agg) | -0.2 pt | 0.24x | 0.60x | 1.24x |
| rf_multi_mod / agg | -0.5 / -1.6 pt | fit in 1 GB (baseline OOM) | no baseline to compare | no baseline to compare |
| xgb_multi_mod / agg | -1.0 / -2.9 pt | 0.27x / 0.074x | 0.24x / 0.08x | 6.0x / 9.1x |
| mlp prune 50 / 70 / 90 % (binary) | +0.2 / 0.0 / -6.4 pt | 1.0x (gzip: see notebook) | ~1.0x | ~1.0x |
| mlp int8 full-integer (binary / multi) | +1.4 / -4.3 pt | 0.12x | 0.003x | 13x / 11x |
| mlp int8 dynamic-range (binary / multi) | 0.0 / -1.0 pt | 0.11x | 0.003x | 8.8x / 8.3x |
| mlp float32 TFLite control (binary) | 0.0 pt | 0.30x | 0.003x | 6.9x |
| small student, distilled / control (binary) | +0.1 / +0.6 pt | 0.19x | 1.08x | 0.81x / 1.13x |
| distilled student + int8, full-integer (binary / multi) | +1.1 / -6.2 pt |
| distilled student + int8, dynamic-range (binary / multi; validation-chosen for 8-class) | +0.1 / -3.6 pt | 0.04x | 0.003x | 31x / 20x |

Observations (each from the table above):
* Tree size and depth limits cut model size by 1-2 orders of magnitude for 0.5-3 pt of macro F1; XGBoost and Random Forest benefit most.
* **Quantisation vs runtime**: the float32 TFLite control reaches almost the same batch-1 latency as int8, so the MLP's huge speed-up is a *runtime* effect (Keras eager overhead vs the TFLite interpreter), not an int8 effect. int8's own gain is the ~8x smaller file and ~1.9x higher batch-1024 throughput over float32 TFLite.
* **Distillation did not help**: the distilled student matches the same-architecture control within noise (`results/distill_search_*.json`, `results/seed_stability.csv`).
* **Unstructured pruning** does not shrink the dense Keras file or speed up dense inference; only the gzip size shrinks (`size_gzip_kb` in `results/metrics_mlp_*_prune*.json`).
* The quantisation mode (full-integer vs dynamic-range) was chosen on validation macro F1 and size only (`results/mlp_quant_choice.json`): full-integer for the binary task, dynamic-range for 8-class. Both are reported.

## 4. Seeds (`results/seed_stability.csv`; seeds 42/43/44, split fixed)
Tree models are stable (std <= 0.0015; XGBoost `hist` is deterministic here, std 0). MLP variants vary more (binary std 0.005-0.019), so MLP-vs-tree and distilled-vs-control gaps below ~1.5 pt should not be trusted.

Models with a single seed only: aggressive tree variants, pruned MLPs, the float32 TFLite control and the distilled+int8 models.

## 5. Verification
`verify.py` checks: split disjointness, sizes and class proportions; no identical feature rows across splits; scaler statistics; exactly one test access per model and model files frozen before it; metrics recomputed from saved predictions to 1e-9; confusion matrices sum to the test size; FPR formula; model sizes equal `os.path.getsize`; benchmark sanity (latency > 0, batch-1024 throughput > batch-1). All pass. No macro F1 exceeds 0.99, but the leakage checks were run anyway (`python verify.py --leak`): the shuffled-label control collapses and no single feature reaches 0.99. Top XGBoost importances: Header_Length, Protocol Type, Time_To_Live, Rate, fin_flag_number (`results/rankings_xgb_*.json`). One bug was found and fixed by `verify.py` (binary FPR used the wrong positive class; DECISIONS.md #9).

## 6. Limitations
* Docker Desktop on a laptop (i5-12500H, x86, inside a VM) is **not** a Raspberry Pi 4: absolute times will differ. Native pinned-core latencies are on average ~1.2x the Docker ones, so rankings, not absolute numbers, are the takeaway.
* Single dataset (CICIoT2023). TON_IoT and N-BaIoT were **not run** (planned work; files not present, nothing downloaded).
* Sampling bias: rare attacks are kept whole while DDoS/DoS are down-sampled, so the class mix differs from real traffic and benign is ~2.8%. Deduplication after sampling (238,729 rows) and removal of identical vectors with conflicting labels (52,116 rows) change the difficulty; scores are not comparable with papers that use the raw data.
* One split and few seeds: variants are compared on one split; the MLP seed spread is up to ~2 pt.
* Peak RAM is dominated by the library runtime (TensorFlow ~560 MB in Docker); the TFLite benchmarks load full TensorFlow. A real gateway would use `tflite-runtime` or compiled trees, lowering RAM. RSS growth during prediction was under 0.5 MB, so "peak RAM" mostly measures the runtime, not the model.
* CPU utilisation is ~100% for every model because the loops are single-threaded and back-to-back; it is not discriminating. Per-prediction CPU cost equals latency.
* Test-set use: every reported model (including seed models and controls) was scored once. The dynamic-range q8 models were first produced when full-integer conversion failed (a bug in my representative-dataset generator), then kept as `*_q8dyn` and new full-integer models were scored separately (DECISIONS.md #10).
* `tree_reduction_grid.csv` size and latency are in-process proxies (300 batch-1 calls); official numbers are the Docker benchmark of the selected variants.
* Pruning used manual masks (not tfmot): one-shot magnitude pruning per layer with 8 fine-tuning epochs.
* `requirements.txt` pins the package versions used. The `*_q8dyn` models now have a code path in `src/reduce_mlp.py`; the ones in `models/` were created earlier by the fallback.
* Two moderate/aggressive pairs (DT binary, DT multi) selected the same configuration because the DT grid is coarse.

## 7. Open questions and future work
* Tree models in Python vs exported C: undecided in the synopsis and **not tested** here; the Python runtime floor (~130 MB RSS, ~0.1 ms per flow) suggests a C export could cut RAM and latency further.
* What RAM limit is acceptable for a gateway: undecided. Here 1 GB was used; the multiclass Random Forest baseline does not fit, all other models do (largest: `rf_multi_mod`, 823 MB).
* Raspberry Pi 4 run, TON_IoT and N-BaIoT replication, structured pruning and more distillation settings are future work.

## 8. Why the false-positive rate is high (validation-only diagnostic)
**Status: read-only diagnostic on the validation split. No model was retrained or changed, the test set was not touched (`test_access.log` unchanged at 80 entries), and no headline number above changes.** Code and outputs: `diagnostics/run_diagnostics.py`, `diagnostics/step3b_confusable_classes.py`, all tables in `diagnostics/`. The threshold results below are validation numbers, not test results, and were not used to pick anything.

**Short answer.** The high FPR (benign flagged as attack, 0.40-0.51) is not caused by cleaning or by exact duplicates. It comes from (a) genuine overlap in feature space between Benign flows and four low-volume attack categories (Recon, Web-based, Brute force, Spoofing), and (b) the default 0.5 operating point, which favours attack recall because Benign is only ~2.8% of the data. Threshold choice moves FPR a lot; the ranking quality (AUC 0.988) stays high, but macro F1 stays within ~0.79-0.83 at any threshold.

1. **Cleaning (`diagnostics/step1_cleaning_balance.csv`, `step1_rows_per_label_per_step.csv`).** Benign 21,329 to 21,293 (only 36 rows removed: 10 exact duplicates, 26 label conflicts). Attack 1,034,556 to 737,621 (238,719 exact duplicates, 52,090 label conflicts, 6,092 same-category identical vectors, 34 missing/NaN). Benign share rose from 2.02% to 2.81%. The conflict removal is almost entirely DDoS vs DoS (23,959 of ~24,100 conflicting vectors, `step1_conflict_category_combos.csv`); Benign-vs-attack conflicts are rare. So cleaning did not take Benign away and, if anything, raised its share.
2. **Benign vs attack on validation (`step2_val_benign_vs_attack.csv`, `step2_cm_val_*.csv`).** Attack F1 is ~0.99 for every model, Benign F1 is only 0.57-0.64:

| model | Benign precision | Benign recall | Benign F1 | Attack recall | FPR |
|---|---|---|---|---|---|
| dt_binary | 0.579 | 0.563 | 0.571 | 0.988 | 0.437 |
| rf_binary | 0.728 | 0.535 | 0.617 | 0.994 | 0.465 |
| xgb_binary | 0.709 | 0.590 | 0.644 | 0.993 | 0.410 |
| mlp_binary | 0.678 | 0.498 | 0.574 | 0.993 | 0.503 |

   (Validation: 3,194 Benign, 110,643 attack rows.) Binary macro F1 is simply the mean of a ~0.99 Attack F1 and a ~0.6 Benign F1.
3. **Separability (`step3_separability.json`, `step3b_*.csv`).** No validation Benign row has an exactly identical vector to a training attack row; after rounding to 2-4 decimals 2 of 3,194 do (0.06%), at integer rounding 9 (0.28%). So the overlap is not duplicates but proximity: in standardised space, 46.2% of validation Benign rows have a nearer training attack row than any training Benign row, versus 0.9% for a same-size sample of validation attack rows; the mean attack share among the 5 nearest training neighbours is 49.8% for Benign and 98.6% for attacks (training data is 97.2% attack). Only 14.1% of Benign rows have all 5 neighbours Benign. With the validation 8-class XGBoost, 29.9% of Benign rows are predicted as an attack category: Recon 20.3%, Web-based 6.0%, Spoofing 1.3%, Brute force 2.2%; none as DDoS, DoS or Mirai. Conversely 11.0% of Brute force, 11.1% of Recon, 4.9% of Spoofing and 6.2% of Web-based flows are predicted Benign. DDoS, DoS and Mirai are essentially perfectly separated from Benign.
4. **Threshold sweep (`step4_*_threshold_sweep.csv`, `step4_*_fpr_at_recall.csv`, `diagnostics/fpr_vs_recall.png`).** XGBoost (depth 10, 300 trees, as the baseline), trained on train, scored on validation. AUC 0.988 for both balanced and unweighted. Unweighted at 0.5 (the baseline setting): FPR 0.410 at attack recall 0.993. Balanced class weights at 0.5: FPR 0.113 at recall 0.968. Balanced, FPR at a given attack recall:

| attack recall | 0.95 | 0.98 | 0.99 | 0.995 |
|---|---|---|---|---|
| FPR | 0.049 | 0.190 | 0.319 | 0.475 |

   Best validation macro F1 over thresholds is 0.832 (at threshold 0.10), against 0.817 for the unweighted model at 0.5, so no threshold lifts macro F1 much. The models' high FPR is mostly the price of ~99% attack recall: the last 5% of attacks look like Benign. Note the baseline tuning compared class-weighted and unweighted models only at threshold 0.5 on macro F1, which is why the unweighted model won there.
5. **Cleaning choice (`step5_*`).** On a variant where only exact same-label duplicates are removed (817,122 rows, Benign 2.61%, label conflicts kept; own 70/15/15 split with seed 42, so not the same rows as the main split), the balanced XGBoost has AUC 0.989 and FPR 0.040 / 0.174 / 0.295 / 0.444 at recall 0.95 / 0.98 / 0.99 / 0.995, i.e. 1-3 points lower FPR than the fully cleaned data (0.049 / 0.190 / 0.319 / 0.475), best validation macro F1 0.835 vs 0.832. The cleaning choice therefore explains little. Caveat: in that variant 5.8% of validation rows (0.03% of Benign rows) also occur in the training set, so it is slightly optimistic, which is part of why the main pipeline removes such rows.

**Implications.** (i) The FPR numbers in sections 2-3 are real properties of this feature set and this operating point, not an artefact of cleaning or leakage. (ii) A deployment would choose the threshold from a validation FPR-vs-recall curve (for example FPR ~5% at ~95% attack recall) instead of 0.5; that was not applied to any reported test number. (iii) The synopsis' "attack vs benign" framing hides that DDoS/DoS/Mirai are easy while Recon, Web-based, Brute force and Spoofing are hard to tell from normal traffic.

**Limits of this diagnostic.** Validation set only, one seed, one model family for the sweeps; nearest-neighbour statistics use Euclidean distance on standardised features and a single random sample of attack rows; the cleaned and exact-dup-only variants use different splits.

## 9. Optimization with GA, PSO and ACO (phase 2; `iot_ids_optimization.ipynb`, `results/optimization/`)
**Setup.** GA, PSO, ACO (ACOR, ACO for continuous domains, an adaptation, not classical discrete ACO) and a random-search control each searched the same encoding (39 feature switches + 5-7 hyperparameters per model) with identical population (20), seed, first 20 candidates and number of unique evaluations (dt 800, rf 800, xgb 800/650, mlp 600/580 for binary/8-class). Fitness = macro F1 on a 50k validation subsample of a model trained on a 100k training subsample, minus 0.001 x feature fraction. Winners were retrained on the full training split and scored once per model id on test (each winner configuration is also retrained and scored with seeds 43 and 44, so it touches test three times; selection never used these scores; every use is in `test_access.log`, now 176 lines). Test numbers below are the seed-42 final models; baselines are the unchanged originals.

| model | task | baseline F1 | GA | PSO | ACO | random | best optimizer vs baseline |
|---|---|---|---|---|---|---|---|
| DT | binary | 0.7854 | 0.7939 | 0.7912 | **0.7957** | 0.7913 | +0.010 (ACO) |
| RF | binary | 0.8021 | 0.8106 | 0.8027 | **0.8120** | 0.7935 | +0.010 (ACO) |
| XGB | binary | **0.8194** | 0.8116 | 0.8160 | 0.8190 | 0.8055 | none beat baseline (ACO -0.0004) |
| MLP | binary | 0.7800 | 0.7657 | 0.7813 | **0.7843** | 0.7719 | +0.004 (ACO), inside seed noise |
| DT | 8-class | 0.7084 | **0.7184** | 0.7164 | 0.7155 | 0.7173 | +0.010 (GA) |
| RF | 8-class | 0.7404 | 0.7465 | 0.7460 | **0.7520** | 0.7339 | +0.012 (ACO) |
| XGB | 8-class | **0.7554** | 0.7481 | 0.7421 | 0.7477 | 0.7373 | none beat baseline (GA -0.007) |
| MLP | 8-class | **0.7103** | 0.6906 | 0.6611 | 0.6873 | 0.6526 | none beat baseline (GA -0.020) |

**What the data show.** (1) Decision tree and random forest improved by +0.6 to +1.2 pt macro F1 for every optimizer in both tasks, except binary RF-PSO (+0.0006), binary RF random search (-0.009) and 8-class RF random search (-0.0065). (2) XGBoost and the 8-class MLP were not improved by any method; the original baselines had already been tuned on the full training set, while the search tuned on 100k-row subsamples, which is a plausible reason but not tested. (3) Binary MLP gains (+0.004 ACO) are smaller than its seed SD (0.011); the 3-seed mean of ACO is 0.7727 vs 0.7755 for the baseline, i.e. no gain. (4) Against random search the best optimizer is ahead by +0.1 to +3.8 pt in all 8 pairs, but for DT random search already captures most of the gain (e.g. 8-class DT: GA 0.7184 vs random 0.7173; ACO and PSO are below random there), so these differences are within what one search seed could produce. No optimizer is consistently best: among the three, ACO has the highest test F1 in 5 of 8 pairs (binary DT/RF/XGB/MLP, 8-class RF), GA in 3 (8-class DT/XGB/MLP), PSO in none. We did not run several search seeds, so we cannot rank the three optimizers statistically. (5) The binary RF gain comes with a different operating point: FPR falls from 0.470 to 0.175-0.196 and recall rises (0.76 to ~0.90) while accuracy and precision drop (0.981 to 0.973, 0.858 to ~0.75); macro F1 only improves slightly. Binary DT FPR got worse (0.421 to 0.446-0.468), the binary XGB and MLP FPRs changed by about -0.03 to +0.04. (6) Seed noise (training randomness only, split fixed): SD 0.0000-0.002 for trees, 0.002-0.011 for MLP.

**Cost.** 32 runs took about 14 h wall (4 runs in parallel, 4 threads each) and 127.6 CPU-hours; unique evaluations 5830 per method (PSO 5719: binary RF-PSO stopped at 689/800 because the swarm collapsed on cached points). Wall time per run (4 runs sharing the CPU): DT 5-17 min, binary XGB 13-21 min, 8-class XGB 1.9-3.1 h, binary RF 0.9-1.3 h, 8-class RF 1.5-3.8 h, MLP 1.8-2.7 h. Total CPU time by method: random 76.9 h, GA 115.3 h, PSO 128.4 h, ACO 138.6 h; the planned 10 h wall budget was exceeded (see DECISIONS.md #23). The best candidate was often found late (e.g. ACO binary DT at eval 800 of 800), so larger budgets might help. Versus the manual tree-reduction grid (section 3): that grid trades accuracy for size (e.g. `rf_binary_mod` 0.7914 with 20 features; `xgb_binary_mod` 0.8122 with 39 features and smaller trees); optimized RF/XGB winners keep similar F1 at 18-29 features, but the optimizer winners are not size-optimized: the optimized 8-class RF winners (ACO/GA/PSO) are OOM at 1 GB like the baseline, binary RF winners are 56-103 MB (baseline 140 MB), XGB 8-class winners are 4.8-13.6 MB (baseline 43.5 MB). The search did not target size, so some (binary XGB ACO 1.9 MB vs 5.5 MB baseline, binary DT 26-65 KB vs 719 KB baseline) are much smaller than the baselines, while RF stays large.

**Resources (Docker, 1 CPU, 1 GB).** Every non-RF-multi model ran; `rf_multi_{aco,ga,pso}` are OOM-killed (exit 137), only `rf_multi_random` (42 MB) fits. Median batch-1 latency: DT 0.09-0.10 ms (baseline 0.16), XGB binary 0.13-0.29 ms (baseline 0.54), MLP 2.0-2.9 ms (baseline 1.9), RF binary 4.0-5.5 ms (baseline 4.5). Baseline rows are from the original benchmark run, not rerun now.

**Limitations.** Fitness is computed on subsamples (100k/50k), a single train/val/test split, one search seed per method, and ACO is the continuous ACOR variant. Final F1 differences below about 0.01 should not be read as method ranking. Selection never saw test, but val-subsample fitness over-estimates (winners are selected for the maximum of hundreds of noisy evaluations).
