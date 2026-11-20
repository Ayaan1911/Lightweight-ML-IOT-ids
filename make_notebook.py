"""Generate iot_ids.ipynb (sections in the required order). Conclusion text comes from results/conclusion.md."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

code("""
import sys, json, glob
from pathlib import Path
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from IPython.display import display, Image, Markdown
from src.common import *
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60); pd.set_option("display.float_format", lambda v: f"{v:.4f}")
def csv(name, **kw): return pd.read_csv(RES / name, **kw)
""")

md("""
# 1. Goal
Detect attacks in IoT network traffic (CICIoT2023) and compare Decision Tree, Random Forest, XGBoost and an MLP on **detection quality (macro F1 is the main metric)** and **resource use** (model size, peak RAM, CPU, batch-1 latency, batch-1024 throughput). Then build reduced versions and compare each with its original.
All code lives in `src/`, `bench.py`, `verify.py`; this notebook only imports it and shows the saved outputs. Numbers shown here are read from the CSV/JSON files in `results/`.
""")

md("## 2. Dataset\nCICIoT2023 `MERGED_CSV/Merged*.csv` (63 files, 39 numeric flow features + `Label`). Used in place, streamed in chunks (never fully loaded).")
code("""
print("Sampling rule (src/data.py): pass A counts rows per label; pass B draws an exact stratified sample (seeded, seed 42, float32).")
print("  * labels with < 30,000 rows in total: ALL rows kept")
print("  * other labels: proportional share of (1,000,000 - rows kept above), with a floor of 10,000 rows (or all rows if fewer)")
display(csv("sampling_counts.csv", index_col=0))
""")
md("The sample keeps rare attacks whole and thins the huge DDoS/DoS classes, so class balance in the sample differs from the real traffic (limitation). Benign is only ~2.8% of the sample. TON_IoT and N-BaIoT are **planned work** (their files are not present locally; nothing was downloaded).")

md("## 3. Inspection")
code("""
print(json.dumps(jload(RES / "inspect_summary.json"), indent=1))
display(csv("inspect_dtypes.csv", index_col=0).value_counts().rename("n_columns").to_frame())
display(csv("inspect_columns.csv", index_col=0).describe().T)
display(csv("inspect_label_counts.csv", index_col=0).T)
""")
md("No IP, port or timestamp columns exist in the Merged files (so none were dropped). A few NaN / infinite values and many duplicate rows were found in the sample; no column is constant.")

md("## 4. Cleaning")
code("""
print((RES / "cleaning_log.md").read_text())
display(csv("clean_class_counts.csv", index_col=0).T)
display(csv("label_mapping.csv").groupby("category").label.apply(list).to_frame())
""")
md("Duplicates are removed **before** the split, then rows whose identical feature vector has conflicting task labels, then remaining identical vectors, so no identical feature row can sit in two splits (checked in `verify.py`). The conflicting-label removal shows that some DoS/DDoS (and benign/attack) flows are indistinguishable with these features.")

md("## 5. Split")
code("""
ix = np.load(PROC / "split.npz"); df = pd.read_parquet(PROC / "clean.parquet", columns=["y_multi", "y_binary"])
print({k: len(ix[k]) for k in ("train", "val", "test")}, "seed", SEED)
display(pd.DataFrame({k: df.y_multi.iloc[ix[k]].value_counts(normalize=True) * 100 for k in ("train", "val", "test")}).round(3).rename_axis("% per class"))
print((RES / "split_note.txt").read_text())
""")
md("One 70/15/15 split stratified on the 8-class label, shared by the binary and multiclass tasks and by every model. The StandardScaler is fitted on train only (`models/scaler.npz`, used by the MLP, distillation and quantization calibration); trees use raw features.")

md("## 6. Baseline training")
code("""
cfg = jload(RES / "baseline_config.json")
display(pd.DataFrame([{"model": k, "chosen": str(v["chosen"]), "val_macro_f1": v["val_macro_f1"], "train_time_s": v["train_time_s"], "n_trials": len(v["trials"])} for k, v in cfg.items()]))
""")
md("Light tuning on the **validation set only** (class-weighted vs unweighted included, time budget per model). Full trial lists are in `results/baseline_config.json`.")

md("## 7. Evaluation (test set, each frozen model scored once)")
code("""
m = pd.DataFrame([jload(f) for f in sorted(glob.glob(str(RES / "metrics_*.json")))]).set_index("model_id")
base = [f"{n}_{t}" for t in ("binary", "multi") for n in ("dt", "rf", "xgb", "mlp")]
display(m.loc[base, ["task", "macro_f1", "accuracy", "precision", "recall", "fpr"]])
display(m.loc[[b for b in base if b.endswith("multi")], [c for c in m.columns if c.startswith("f1_")]])
for b in ("xgb_binary", "xgb_multi"):
    display(Markdown(f"confusion matrix `{b}` (csv + png for every model in results/ and figures/)")); display(csv(f"cm_{b}.csv", index_col=0))
display(Image(str(FIG / "cm_xgb_multi.png")))
""")
md("FPR: binary = FP/(FP+TN) with *attack* as positive (benign flows flagged as attack); multiclass = macro-averaged one-vs-rest. Predictions of every model are saved (`results/pred_*.npz`), and `verify.py` recomputes all metrics from them.")

md("## 8. Resource measurements")
code("""
cols = ["model_id", "size_kb", "peak_rss_mb", "peak_rss_increase_mb", "cpu_util_pct", "lat_mean_ms", "lat_median_ms", "lat_p95_ms", "lat_p99_ms", "throughput_b1024_per_s", "oom"]
for tag, title in (("docker", "OFFICIAL: Docker --cpus=1 --memory=1g (one container per model)"), ("native", "SECONDARY: native, not Docker-limited, pinned to one core")):
    b = csv(f"benchmark_{tag}.csv"); display(Markdown(f"**{title}**"))
    display(b[[c for c in cols if c in b]][b.model_id.isin(base)].set_index("model_id"))
""")
md("Each model runs in its own fresh process/container, 1 thread everywhere, 20,000 saved test rows as input, 200 warm-ups, 10,000 single-flow predictions, ≥25 batch-1024 repeats. A model that exceeds the 1 GB limit is recorded as OOM, not re-run with more memory. Docker on a laptop is not a Raspberry Pi.")

md("## 9. Model reduction")
code("""
g = csv("tree_reduction_grid.csv"); print("grid rows:", len(g), "(results/tree_reduction_grid.csv)")
display(csv("tree_selected_variants.csv", index_col=0)[["k", "rank", "depth", "n_est", "val_macro_f1", "baseline_val_f1", "size_kb", "lat_ms"]])
d = jload(RES / "mlp_quant_choice.json"); display(pd.DataFrame({k: {"chosen": v["chosen"]} for k, v in d.items()}).T)
mm = m[m.index.str.startswith("mlp_") & ~m.index.str.contains("_s4")]
display(mm[["task", "macro_f1", "val_macro_f1", "sparsity", "size_raw_kb", "size_gzip_kb", "quant_mode", "T", "alpha"]].sort_index())
for t in ("binary", "multi"): print(t, "distillation search:", json.dumps(jload(RES / f"distill_search_{t}.json")))
""")
md("""
Trees: top-k features (mutual information vs model importance), depth limits and fewer trees; *moderate* = smallest variant within 1 point of the baseline **validation** macro F1, *aggressive* = within 3 points. MLP: magnitude pruning 50/70/90% (manual masks + fine-tuning; unstructured sparsity does **not** shrink the dense file or speed up dense inference — only the gzip size drops), TFLite int8 (full-integer chosen vs dynamic-range on validation F1 and size only), distillation with a control student of identical architecture, and distilled+int8.
""")

md("## 10. Final comparison")
code("""
for tag in ("docker", "native"):
    display(Markdown(f"**Baselines — {tag}** (`results/baseline_comparison_{tag}.csv`)")); display(csv(f"baseline_comparison_{tag}.csv").set_index("model").T)
r = csv("reduced_vs_original_docker.csv")
display(Markdown("**Reduced vs original — Docker** (`results/reduced_vs_original_docker.csv`)"))
display(r.sort_values(["task", "family", "variant"]).set_index("model_id"))
display(Markdown("**Macro F1 over seeds 42/43/44** (`results/seed_stability.csv`; split fixed; XGBoost hist is deterministic so std=0)")); display(csv("seed_stability.csv"))
""")
code("""
for t in ("binary", "multi"):
    for k in ("size", "latency"): display(Image(str(FIG / f"pareto_docker_{t}_{k}.png")))
""")
md((ROOT / "results" / "conclusion.md").read_text() if (ROOT / "results" / "conclusion.md").exists() else "## Conclusion\n(pending)")

nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, ROOT / "iot_ids.ipynb")
