# Methods and experimental setup specification

Sources read: `src/training.py`, `src/reduce_trees.py`, `src/reduce_mlp.py`, `src/seeds.py`, `src/quant_choice.py`, `src/tflite_f32.py`, `src/predictors.py`, `src/evaluation.py`, `src/tables.py`, `bench.py`, `run_bench.py`, `run_bench_docker.py`, `run_docker.ps1/.sh`, `optimization/bench_docker.py`, `optimization/tables.py`, `optimization/make_notebook.py`, `Dockerfile`, `requirements.txt`, `results/baseline_config.json`, `results/tree_reduction_grid.csv`, `results/tree_selected_variants.csv`, `results/distill_search_*.json`, `results/mlp_quant_choice.json`, `results/seed_stability.csv`, `results/optimization/optimization_seed_stability.csv`, `results/benchmark_*.csv`, `REPORT.md`, `DECISIONS.md`, `PROGRESS.md`, `README.md`, `make_notebook.py`, `.venv/pyvenv.cfg` and the installed package metadata of `.venv`.
Nothing was re-run or modified. "NOT FOUND" means it is not in the code or logs.

Common to everything: one shared 70/15/15 split (seed 42); features are the 39 cleaned columns (see `PREPROCESSING_SPEC.md`); the training split is used for fitting, the validation split for every choice, and each model id is scored on the test split once (`test_access.log`). Binary decision rule: sigmoid/probability > 0.5; 8-class: argmax.

## 1. Baselines

### 1.1 Constructor arguments (everything not listed is the library default)

| Model | Arguments set in code | Chosen values (`baseline_config.json`) |
|---|---|---|
| Decision Tree (`sklearn.tree.DecisionTreeClassifier`) | `random_state=42`, `max_depth`, `class_weight` | binary: `max_depth=20`, `class_weight=None`; 8-class: `max_depth=20`, `class_weight=None` |
| Random Forest (`sklearn.ensemble.RandomForestClassifier`) | `random_state=42`, `n_jobs=-1`, `n_estimators=100`, `class_weight` | binary: `class_weight=None`; 8-class: `class_weight="balanced_subsample"` |
| XGBoost (`xgboost.XGBClassifier`) | `tree_method="hist"`, `n_jobs=-1`, `random_state=42`, `max_depth`, `n_estimators`, `learning_rate`; optional balanced `sample_weight` in `fit` | binary and 8-class: `max_depth=10`, `n_estimators=300`, `learning_rate=0.1`, no sample weights |
| MLP (Keras 3, `Sequential`) | see 1.2 | binary and 8-class: hidden (128, 64), no class weights |

Defaults that matter (library defaults, not set in the code; stated from the code's absence of an override, not read from library source): DT: Gini criterion, `min_samples_split=2`, `min_samples_leaf=1`, `max_features=None` (all 39 features at every split). RF: `max_depth=None` (fully grown trees), `bootstrap=True`, `max_features` default, `criterion` default. XGBoost: `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, regularisation and `max_bin` all left at defaults; objective chosen automatically from the label type (binary or multiclass). NOT FOUND: the numeric default values themselves are not written in any project file.

### 1.2 MLP architecture and training (`training.build_mlp`, `fit_mlp`)

* Inputs: 39 features standardised with a `StandardScaler` fitted on the training split only (`models/scaler.npz`); the scaling is done outside the network, in `predictors.py`.
* Layers: `Dense(128, relu)` then `Dense(64, relu)`; no dropout, no batch normalisation. Output: `Dense(1, sigmoid)` (binary) or `Dense(8, softmax)` (8-class).
* Loss: `binary_crossentropy` or `sparse_categorical_crossentropy`. Optimizer: `"adam"` with the Keras default learning rate (no explicit value in the code). Batch size 512.
* Epochs: at most 30, with `EarlyStopping(patience=3, restore_best_weights=True)`; monitor not set (Keras default, validation loss) on the validation split.
* Seed: `keras.utils.set_random_seed(42)`. Class weights: none in the chosen models (see 1.3).

### 1.3 Was anything tuned, and were class weights used?

* Yes, a small validation-only search (`training.run_baselines`, budget `BUDGET_S = 420` s per model and task; every candidate fitted fully, since all fit in the budget). Training split for fitting; **validation macro F1** to choose (highest wins); test never used.
  * DT: 6 candidates, `class_weight` in {None, balanced} x `max_depth` in {None, 30, 20}.
  * RF: 2 candidates, `class_weight` in {None, balanced_subsample}, 100 trees.
  * XGBoost: 8 candidates: (unweighted, balanced sample weights) x (depth, trees, lr) in {(6,150,0.2), (10,150,0.2), (10,300,0.1), (14,200,0.1)}.
  * MLP: 2 candidates: with or without balanced class weights, architecture fixed at (128, 64).
* Class weights were **tried** for every baseline, and **selected** only for the 8-class Random Forest (`balanced_subsample`). All other chosen baselines are unweighted (e.g. binary XGB unweighted 0.8172 vs weighted at most 0.8071; binary MLP 0.7816 vs weighted 0.6846 validation macro F1).
* The seed for all baselines is 42.

## 2. Tree reduction (`src/reduce_trees.py`)

* **Data**: training split to fit; validation split to score (macro F1) and to select. Test used only afterwards to score the two selected variants per model once.
* **Rankings** (computed per model and task): (a) `mi`: `mutual_info_classif` on a random 100,000-row training subsample (`default_rng(42)`, `random_state=42`); (b) `imp`: feature importance of the baseline model (`feature_importances_` for DT/RF, XGBoost gain from the booster).
* **Grid values**: k (number of top-ranked features) in {all (39), 30, 20, 15, 10, 5}; depth limits DT {None, 20, 15, 10, 8, 5}, RF {None, 12, 8, 5}, XGB {20, 12, 8, 5}; number of trees (RF, XGB) {100, 50, 25, 10}. Other arguments are fixed at the baseline's chosen values: DT/RF `class_weight`; XGB `learning_rate` and sample-weight choice. Seed 42.
* **Variants per model and task** (nominal): DT: 2 rankings x 6 k x 6 depths = 72. RF and XGB: a ranking ablation of 2 rankings x 6 k at the baseline-sized setting (first depth in the list, 100 trees) = 12, then the ranking with the higher mean validation macro F1 in the ablation gets the full grid 6 k x 4 depths x 4 tree counts = 96, giving 108.
* **The "576"**: 2 tasks x (72 + 108 + 108) = **576 nominal variants**. **The saved CSV `results/tree_reduction_grid.csv` has 552 rows**, because for RF and XGB six ablation settings per model and task coincide with cells of the full grid (the same k, ranking, depth 100-tree setting) and are skipped as already seen: DT 72 + RF 102 + XGB 102 per task = 276, x 2 tasks = 552. Chosen ranking: `imp` for RF (both tasks), `mi` for XGB (both tasks); both rankings for DT.
* **Measures per variant**: validation macro F1; size (in-memory `joblib.dump` bytes for DT/RF, raw JSON booster bytes for XGB, in KB); in-process batch-1 latency (median of 300 calls after 30 warm-up calls, one thread). These two are proxies; the Docker benchmark is the official measurement.
* **Selection rule** (`select`): baseline validation macro F1 `b`. Moderate: among all grid variants of that model and task with validation macro F1 >= `b - 0.01`, take the smallest `size_kb`, ties broken by the lowest `lat_ms`. Aggressive: same with tolerance `0.03`. If nothing qualifies, the variant with the best validation F1 is used (fallback in code; not triggered for any selected variant as the saved values show F1 within tolerance).
* Selected variants (`tree_selected_variants.csv`): binary DT: all features, depth 5 (moderate = aggressive); 8-class DT: all features, depth 15 (moderate = aggressive); binary RF moderate: top 20 (`imp`), depth 12, 25 trees; binary RF aggressive: top 30, depth 12, 10 trees; 8-class RF moderate: top 30, depth None, 25 trees; aggressive: top 30, depth None, 10 trees; binary XGB moderate: all features, depth 8, 100 trees; aggressive: top 20 (`mi`), depth 8, 25 trees; 8-class XGB moderate: all features, depth 12, 50 trees; aggressive: all features, depth 8, 50 trees.

## 3. MLP reduction (`src/reduce_mlp.py`)

All reductions start from the baseline MLP of the same task (128, 64) and use the scaled training/validation data.

### 3.1 Pruning
* Unstructured magnitude pruning, **per layer** (not global): for every `Dense` layer (including the output layer) the threshold is the `sparsity` quantile of |w| of that layer's weight matrix; biases are not pruned.
* Sparsity levels 50%, 70%, 90%.
* Fine-tuned afterwards: 8 epochs, Adam (default lr), batch 512, validation data supplied, no early stopping; the masks are re-applied after every training batch and once at the end. Manual masks, not `tensorflow_model_optimization` (`DECISIONS.md` #8). Seed: pruned models single run (the base weights come from the seed-42 baseline).
* Reported sparsity = mean over layers of the actual zero fraction; sizes: raw `.keras` file and gzip level 9.

### 3.2 Quantisation (TensorFlow Lite; converter `from_keras_model`, `Optimize.DEFAULT`)
* **Full-integer** (`mlp_*_q8`, `mlp_*_distq8`): `representative_dataset` = 500 random training rows (standardised, `default_rng(42)`), `supported_ops = TFLITE_BUILTINS_INT8`; input and output types are not forced to int8 (not set in the code, so the model keeps float32 I/O). Recorded mode in the metrics files is `full-integer` for all four.
* **Dynamic range** (`mlp_*_q8dyn`, `mlp_*_distq8dyn`): `Optimize.DEFAULT` with no representative dataset (`dynamic=True`). They were first produced by the fallback path when full-integer conversion failed (`DECISIONS.md` #12) and later regenerated by `to_tflite(..., dynamic=True)`.
* **Float32 control** (`mlp_*_tflite_f32`, `src/tflite_f32.py`): TFLite conversion without any optimisation or quantisation.
* **Choice between full-integer and dynamic range** (`src/quant_choice.py`): by validation macro F1 rounded to two decimals (a gap of at least 0.5 point decides), otherwise the smaller file. Result (`mlp_quant_choice.json`): `mlp_binary_q8` and `mlp_binary_distq8`: full-integer; `mlp_multi_q8` and `mlp_multi_distq8`: dynamic-range (validation macro F1 0.6990 vs 0.6703, and 0.6750 vs 0.6488, for dynamic-range vs full-integer). Binary validation macro F1: q8 0.7943 (full-integer) vs 0.7817 (dynamic); distq8 0.7898 vs 0.7795.
* The distilled student quantised is the distilled (not the control) student.

### 3.3 Distillation
* **Teacher**: MLP (256, 128, 64), ReLU, trained with `fit_mlp` (Adam default lr, batch 512, max 30 epochs, early stopping patience 3 on validation loss, no class weights), seed 42. Validation macro F1: binary 0.7856, 8-class 0.7039.
* **Student**: (32, 16) ReLU, output logits then sigmoid/softmax.
* **Loss**: `(1 - alpha) * hard + alpha * T^2 * soft`. Hard = cross-entropy with the true labels. Soft = binary cross-entropy between `sigmoid(teacher_logit / T)` and `sigmoid(z / T)` (binary) or KL divergence between `softmax(teacher_logits / T)` and `softmax(z / T)` (8-class). Teacher logits are computed from teacher probabilities clipped to [1e-6, 1 - 1e-6] on the training split.
* **Grid** (chosen on validation macro F1): (T, alpha) in {(2, 0.5), (4, 0.5), (4, 0.9)}. Best for both tasks: T = 2, alpha = 0.5 (validation macro F1 0.7804 binary, 0.6845 8-class).
* **Training**: max 30 epochs, batch 512, Adam default lr, `EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)` on the **training loss** (no validation data in training), seed 42.
* **Control student**: same (32, 16) architecture and the same training code path with `T = 1`, `alpha = 0` (hard labels only). Validation macro F1 0.7837 (binary), 0.6877 (8-class), so the distilled student did not beat the control on validation.

## 4. Benchmark protocol

* **Docker image** (`Dockerfile`): `FROM python:3.11-slim` (tag only; exact patch version and digest NOT FOUND), pip-installed `numpy==2.4.6 scikit-learn==1.9.1 xgboost==3.2.0 tensorflow-cpu==2.21.0 keras==3.15.1` plus unpinned `psutil` and `joblib`. Only `bench.py`, `src/__init__.py`, `src/common.py` and `src/predictors.py` are copied in.
* **Run flags** (`run_bench_docker.py`, `optimization/bench_docker.py`): `docker run --rm --cpus=1 --memory=1g --memory-swap=1g`, `models/` and `results/bench_input.npy` mounted read-only, output written to a mounted folder; one container, hence one fresh Python process, per model. Official numbers are the Docker ones; a secondary native run (`run_bench.py`) uses a fresh process per model pinned with `psutil.cpu_affinity([2])`, not Docker-limited.
* **Threads**: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `TF_NUM_INTEROP_THREADS`, `TF_NUM_INTRAOP_THREADS` all set to 1; sklearn `n_jobs=1`, XGBoost `nthread=1`, TFLite `num_threads=1`.
* **Input**: `results/bench_input.npy`, the first 20,000 rows of the test split (feature matrix only, float32; `README.md`); used for timing only.
* **Warm-up**: 3 calls of a 1,024-row batch, then 200 batch-1 predictions.
* **Batch-1 latency**: 10,000 single-row calls (`X[i:i+1]`, rows 0..9,999 of the input), each timed with `time.perf_counter_ns` around the full `predict` call (which includes column selection and, for MLPs, standardisation). Reported: mean, median, p95, p99; the tables use the median.
* **Batch-1024 throughput**: windows of 1,024 consecutive rows; loop runs at least 25 repetitions and at least 2 s; each repetition timed with `perf_counter`; throughput = 1,024 / time; the median over repetitions is reported (`n_reps_b1024` is saved).
* **Peak RAM**: `peak_rss_mb` = the process peak resident set (`resource.getrusage(...).ru_maxrss` on Linux, i.e. in Docker; `peak_wset` on Windows native); includes the Python and library runtime. Also saved: RSS before load, after load, during prediction (sampled every 500 batch-1 calls and at each batch-1024 call) and the increase.
* **CPU use**: `psutil` user+system CPU time divided by wall time over the batch-1 loop, the batch-1024 loop and both together (`cpu_util_pct`).
* **Model size**: bytes of the model file(s) in `models/` (`predictors.size_kb`) plus `scaler.npz` for MLP models. Formats: DT and RF `.joblib`; XGBoost `.json` (booster); MLP `.keras`; quantised and float32 control MLPs `.tflite`; the tree-grid size column is only a proxy (section 2).
* **Repetitions**: each model was benchmarked **once** per environment (the runners skip a model whose result file already exists; no repeat or aggregate over several benchmark runs is in the code). Phase 1 CSVs: 40 model ids each for Docker and native; the optimizer winners (32 seed-42 models) were benchmarked once more in Docker by `optimization/bench_docker.py`.
* **Out-of-memory handling**: if no result file is produced, a JSON `{"failed": true, "oom": ..., "returncode": ..., "stderr": ...}` is stored, with `oom = true` when the return code is 137 or -9 or "Killed" is in stderr; an in-process `MemoryError` writes `{"oom": true}` and exits with code 3. Rows with failed/oom stay in the CSV (e.g. `rf_multi` baseline and `rf_multi_{aco,ga,pso}` are OOM at 1 GB).

## 5. Environment

| Item | Value | Source |
|---|---|---|
| Host CPU | Intel Core i5-12500H, 12 cores / 16 threads | `REPORT.md`, `PROGRESS.md`, `README.md` |
| RAM | 15.6 GB | `REPORT.md` |
| OS | Windows 11 Home (platform string `Windows-10-10.0.26300-SP0` from the venv Python) | `PROGRESS.md`, venv check |
| Docker | Docker Desktop, docker 29.9.2; engine VM with 16 CPUs and 7.57 GiB; backend type (WSL 2 or Hyper-V): NOT FOUND | `DECISIONS.md` #13, `REPORT.md` |
| Python (host venv) | The venv's own interpreter reports 3.11.1 (`pyvenv.cfg`); `REPORT.md`/`PROGRESS.md` state 3.11.9. **These disagree; the cause is NOT FOUND**, so state only "Python 3.11". | `.venv/pyvenv.cfg`, docs |
| Python (Docker) | `python:3.11-slim`, patch version NOT FOUND | `Dockerfile` |
| scikit-learn | 1.9.1 (host and Docker) | requirements, Dockerfile, installed metadata |
| XGBoost | 3.2.0 (host and Docker) | same |
| TensorFlow | 2.21.0 (host `tensorflow`; Docker `tensorflow-cpu`) | same |
| Keras | 3.15.1 (host and Docker) | same |
| TFLite runtime | the TFLite interpreter inside TensorFlow (`tf.lite.Interpreter`, `tensorflow-cpu==2.21.0`); the separate `tflite-runtime` package is **not** used or installed | `predictors.py`, Dockerfile |
| NumPy | 2.4.6 (host and Docker) | same |
| pandas | 3.1.1 (host only; not installed in the Docker image) | requirements, installed metadata |
| Other | psutil 7.2.2, joblib 1.6.0 (host); Docker image installs psutil and joblib unpinned, so their versions there are NOT FOUND | requirements, Dockerfile |

## 6. Seeds and seed stability

* **Seed 42**: all baselines, the tree grid, the reduced variants, pruned, quantised, distilled and control MLPs, the teacher, the data split and sampling, the scaler, the optimizer fitness model fits and the seed-42 final optimizer models.
* **Retrained with seeds 42, 43, 44** (split fixed; seed changes only model randomness): baselines DT, RF, XGB, MLP; moderate tree variants (`dt/rf/xgb_*_mod`); the chosen-quantisation MLP (`mlp_*_q8`, conversion of the seed-specific baseline, same mode as chosen); distilled student and control student (`mlp_*_dist`, `mlp_*_ctrl`) (`src/seeds.py`, results in `seed_stability.csv`); and, for the optimization phase, every winner configuration of GA, PSO, ACO and random search, in all 8 model/task pairs, i.e. 32 configurations x 3 seeds (`finalize.py`, `optimization_seed_stability.csv`; 96 final models).
* **Single seed only**: aggressive tree variants, pruned MLPs, the float32 TFLite control, distilled+int8 models and the dynamic-range alternatives (`REPORT.md` section 4). Search seed: one per optimizer run.
* **Seed SD**: the **sample standard deviation (ddof = 1) of test macro F1 over the three seeds 42, 43 and 44** (`np.std(..., ddof=1)`, `seed_stability.csv`, `optimization_seed_stability.csv`).
* **"2 x seed SD" noise rule** (defined in `optimization/make_notebook.py`, used for the optimization comparison): for a model and task, `sd` = the larger of the baseline's seed SD and the winning optimizer's seed SD; a macro-F1 gain `d` is called indistinguishable from seed noise if `d < 2 x sd`, and clearly larger otherwise. The same `2 x sd` threshold is used to flag differences between the best optimizer and random search. This SD covers training randomness only, not split-to-split variation. For phase 1 (reduction) no formula is coded; `REPORT.md` section 4 only states in words that gaps below about 1.5 points between MLP variants should not be trusted.

## 7. Variants in the final results tables and how they were chosen

* **Baselines**: the 8 models selected by validation macro F1 (section 1.3).
* **Tree variants**: one moderate and one aggressive variant per tree model and task (12 ids), each chosen by the rule in section 2. Moderate = smallest variant within 1 point of baseline validation macro F1; aggressive = within 3 points.
* **MLP variants**: all in the tables: pruned 50/70/90%; full-integer and dynamic-range int8 of the baseline; float32 TFLite control; distilled student (T = 2, alpha = 0.5, chosen on validation among 3 settings) and control student; distilled + int8 in both modes. The quantisation mode shown in the headline rows is the validation-chosen one (full-integer for binary, dynamic-range for 8-class); both are listed. Nothing in the MLP list was filtered by test results.
* **Optimizer winners**: for each of 8 model/task pairs x 4 methods, the candidate with the highest search fitness (`DECISIONS.md` #20), retrained on the full training split and scored once on test; see `optimization/ALGORITHM_SPEC.md`.
* **Tables**: `reduced_vs_original_docker.csv` and `_native.csv` list every model id except the `_s43`/`_s44` seed models, with the change in macro F1 relative to the baseline of the same family and task and the ratios of benchmark metrics. 40 model ids appear in the benchmark tables (`all_model_ids.txt`).
