# PROGRESS

## Machine
Windows 11 Home, Intel i5-12500H (12 cores / 16 threads), 15.6 GB RAM, D: ~62 GB free at start, Python 3.11.9. Docker Desktop 29.9.2 works (docker.exe at `C:\Users\ASUS\AppData\Local\Programs\DockerDesktop\resources\bin`, not on PATH); VM 16 CPUs / 7.57 GiB. Caches on `D:\cache`.

## Complete (all 19 steps)
1. Orient, venv, requirements.txt (TensorFlow installed fine on 3.11, no PyTorch fallback).
2. Layout and notebook `iot_ids.ipynb` (sections in required order; executed top to bottom without errors).
3-6. Sample (1,055,885 rows), inspect, clean (758,914 rows), shared 70/15/15 split.
7-8. Four baselines, binary and 8-class; test metrics, predictions, confusion matrices.
9-11. `bench.py`; official Docker benchmark (`--cpus=1 --memory=1g`) for 40 models plus native pinned-core secondary; baseline table and plots.
12-14. Tree reduction grid (552 variants), moderate/aggressive variants; MLP pruning, int8, distillation + control, distilled+int8, float32-TFLite control; reduced-vs-original table, Pareto plots, 3-seed stability.
15. `REPORT.md`.
16. `verify.py`: 1,228 checks pass (including `--leak`).
17. Notebook executed with nbconvert.
18. Independent audit by a fresh subagent: no blockers; reproducibility fixes applied (see `CHECKLIST.md`).
19. README.

## Headline results (Docker, test)
Binary macro F1: XGBoost 0.819, RF 0.802, DT 0.785, MLP 0.780. 8-class: XGBoost 0.755, RF 0.740, MLP 0.710, DT 0.708. Multiclass RF baseline (1.4 GB) is OOM at 1 GB. `xgb_binary_mod`: -0.7 pt F1, 0.14x size, 0.27x latency. `rf_binary_mod`: -1.1 pt, 0.023x size. MLP: pruning and distillation gave no benefit; the int8/TFLite speed-up is mostly a runtime effect (float32 TFLite control).

## Deviations from the synopsis / task
* Docker Desktop is not a Raspberry Pi 4 (x86 laptop VM). No Pi run.
* TON_IoT and N-BaIoT not run (files not present; not downloaded). Planned work.
* Pruning via manual masks, not tensorflow_model_optimization (DECISIONS.md #8).
* Tree grid latency/size are in-process proxies; official numbers only for selected variants.
* Mapping label -> category is the standard CICIoT2023 grouping (README.pdf has no table).
* Two extra controls added: `*_tflite_f32`, and `*_q8dyn` kept next to full-integer `*_q8`.

## Diagnostic (post-hoc, validation only)
* `diagnostics/` + REPORT.md section 8 explain the high binary FPR (class overlap with Recon/Web-based/Brute force/Spoofing and the 0.5 operating point; not cleaning, not leakage). No model, result file or test number was changed; test_access.log had 80 entries at that point (176 after phase 2).

## Open issues
* CPU utilisation is ~100% for all models, so it carries no information (documented in REPORT.md).
* MLP peak RAM measures the TensorFlow runtime, not the model.
* Bug history: binary FPR positive class (fixed, verify.py caught it); TFLite representative dataset generator (fixed); both logged in DECISIONS.md.

## Phase 2: metaheuristic optimization (GA / PSO / ACO / random) - complete
* 32 optimizer runs (4 models x 2 tasks x 4 methods, equal budgets, ~14 h wall, 127.6 CPU-h); 96 final models (seeds 42/43/44) each scored once on test (`test_access.log` now 176 lines); 32 seed-42 models benchmarked in Docker (3 x rf_multi OOM at 1 GB, recorded).
* Code `optimization/`, outputs `results/optimization/`, notebook `iot_ids_optimization.ipynb` (+ pointer cell in `iot_ids.ipynb`), `REPORT.md` section 9, `DECISIONS.md` #16-24.
* Result: DT and RF improve by about +1 pt macro F1; XGBoost (both tasks) and the 8-class MLP are not improved by any method; binary MLP inside seed noise; random search gets most of the DT gain.
* `verify.py`: 2026 checks pass (first attempt), incl. unchanged pre-existing files (hashes), equal budgets, one test use per final model.
* Incidents: iot-bench image had vanished and the first rebuild failed when Docker Desktop crashed under memory pressure (rebuilt later); background monitors were reaped under memory pressure; PSO on binary RF stopped at 689/800 unique evaluations (swarm collapse).
