# Lightweight Machine Learning for Intrusion Detection in IoT Networks: Resource-Aware Model Reduction and Metaheuristic Optimization on CICIoT2023

Mohammad Ayaan (1MS24CY036) and Snehal Parasar (1MS24CY060), M. S. Ramaiah Institute of Technology.

- Four model families (Decision Tree, Random Forest, XGBoost, MLP) on CICIoT2023, binary and 8-class, with model-reduction methods compared against each original.
- Hyperparameter and feature-subset search with GA, PSO and ACO, using random search as the control.
- Every model benchmarked in Docker with 1 CPU and 1 GB of memory.

## Repository layout

| Path | Content |
|---|---|
| `src/` | pipeline code: data, training, tree and MLP reduction, seeds, evaluation, figure script |
| `optimization/` | GA / PSO / ACO / random-search code, orchestration, Docker benchmark, `ALGORITHM_SPEC.md` |
| `results/` | phase 1 and optimization results (CSV, JSON, predictions, logs) |
| `diagnostics/` | FPR diagnostics and cleaning-step diagnostics |
| `figures/` | confusion matrices and other figures, including `convergence.png` / `.svg` |
| `iot_ids.ipynb`, `iot_ids_optimization.ipynb` | project notebooks |
| `bench.py`, `run_bench.py`, `run_bench_docker.py`, `run_docker.sh`, `run_docker.ps1`, `Dockerfile` | benchmark scripts and image |
| `verify.py`, `make_notebook.py` | verification script and notebook builder |
| `REPORT.md`, `DECISIONS.md`, `PROGRESS.md`, `CHECKLIST.md`, `project_steps.md` | report and project notes |
| `PREPROCESSING_SPEC.md`, `METHODS_SPEC.md`, `RESULTS_PACK.md` | exact specifications and result tables for the paper |

## Not in this repository

- **Data:** the CICIoT2023 dataset (`MERGED_CSV/`, sampled and cleaned copies) is not included. Download it from the University of New Brunswick Canadian Institute for Cybersecurity CICIoT2023 page and put the `Merged*.csv` files in `MERGED_CSV/`.
- **Models:** the saved models (`models/`) are not included. Regenerate them with the commands in the "Reproduce" section below.

## Environment

Python 3.11 in the container; scikit-learn 1.9.1, XGBoost 3.2.0, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.4.6.

---

# IoT intrusion detection: lightweight ML (CICIoT2023)

Compare Decision Tree / Random Forest / XGBoost / MLP on detection (macro F1) and resource use, then reduce each and compare with its original. Read `REPORT.md` for results, `iot_ids.ipynb` for the project record, `DECISIONS.md` for every judgement call.

## Reproduce (Windows, Python 3.11; approx. 5 h total on an i5-12500H, 16 GB; the tree grid is the longest step)
Put the 63 `Merged*.csv` files in `MERGED_CSV/`. Keep caches on D: if you like (`PIP_CACHE_DIR`, `TMP`, `TEMP`).
```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt  # pinned versions used for this run
.venv\Scripts\python -m src.data sample        # ~15 min  sampling (pass A + B), writes data_processed/sample.parquet
.venv\Scripts\python -m src.data inspect       # inspection tables
.venv\Scripts\python -m src.data clean         # cleaning + results/cleaning_log.md
.venv\Scripts\python -m src.data split         # shared 70/15/15 split
.venv\Scripts\python -c "from src import data,training;import numpy as np;training.get_scaler();X,_=data.load('test','binary');np.save('results/bench_input.npy',X.iloc[:20000].to_numpy(np.float32))"
.venv\Scripts\python -m src.training           # ~25 min  baselines + test scoring
.venv\Scripts\python -m src.reduce_trees grid  # ~2-3 h   tree reduction grid (resumable)
.venv\Scripts\python -m src.reduce_trees select
.venv\Scripts\python -m src.reduce_trees final # train selected variants, score once
.venv\Scripts\python -m src.reduce_mlp         # ~20 min  prune / int8 / distil / control
.venv\Scripts\python -m src.quant_choice       # q8dyn vs full-integer, validation only
.venv\Scripts\python -m src.tflite_f32         # float32 TFLite control
.venv\Scripts\python -m src.seeds              # ~25 min  seeds 43, 44
```
Benchmarks (ids = every `models/*.meta.json` name without the suffix, listed in `results/all_model_ids.txt`):
```
# official: Docker, --cpus=1 --memory=1g, one container per model (Docker Desktop must be running)
docker build -t iot-bench .
.venv\Scripts\python run_bench_docker.py <ids...>      # -> results/benchmark_docker.csv
# secondary: native, pinned to one core
.venv\Scripts\python run_bench.py native <ids...>      # -> results/benchmark_native.csv
```
Then:
```
.venv\Scripts\python -m src.tables             # comparison tables + Pareto figures
.venv\Scripts\python make_notebook.py          # builds iot_ids.ipynb (conclusion from results/conclusion.md)
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace iot_ids.ipynb
.venv\Scripts\python verify.py [--leak]
```
Every expensive step caches its output and is skipped on re-run if it exists (delete the file to force a rerun). `run_docker.ps1` wraps build + Docker benchmark.

## Phase 2: metaheuristic optimization (GA, PSO, ACO + random control)
Needs the finished phase 1 outputs. About 14 h wall on the i5-12500H; keep caches on D:.
```
.venv\Scripts\python -m optimization.orchestrate probe     # timing probe -> results/optimization/probe/
# write results/optimization/budgets.json (see DECISIONS.md #19), then:
.venv\Scripts\python -m optimization.orchestrate run       # 32 runs, resumable, results/optimization/runs/
.venv\Scripts\python -m optimization.finalize              # retrain winners (seeds 42-44), score once on test
docker build -t iot-bench . ; .venv\Scripts\python -m optimization.bench_docker   # --cpus=1 --memory=1g
.venv\Scripts\python -m optimization.tables ; .venv\Scripts\python -m optimization.make_notebook
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace iot_ids_optimization.ipynb iot_ids.ipynb
.venv\Scripts\python verify.py
```
`optimization/snapshot.py` hashed every phase-1 artifact before this phase so `verify.py` can prove none changed. Results: `REPORT.md` section 9 and `iot_ids_optimization.ipynb`.

## Layout
`src/` code, `bench.py` single-model benchmark, `run_bench*.py` orchestrators, `verify.py` checks, `results/` tables/predictions/metrics, `models/` saved models + `*.meta.json`, `figures/`, `diagnostics/` (post-hoc validation-only FPR diagnostic), `data_processed/` (parquet, split indices). `test_access.log` records every test-set scoring.
