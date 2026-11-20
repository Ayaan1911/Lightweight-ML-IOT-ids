"""Build the CSV tables of the optimization phase (all from saved metrics/runs; no test access).
python -m optimization.tables"""
import glob, numpy as np, pandas as pd
from src.common import ROOT, RES, jload

OUT = RES / "optimization"; FIN = OUT / "final"
MODELS, TASKS, OPTS = ("dt", "rf", "xgb", "mlp"), ("binary", "multi"), ("ga", "pso", "aco", "random")
COLS = ["macro_f1", "accuracy", "precision", "recall", "fpr"]


def metrics_table():
    rows = []
    for t in TASKS:
        for m in MODELS:
            b = jload(RES / f"metrics_{m}_{t}.json")
            rows.append({"model_id": b["model_id"], "task": t, "model": m, "optimizer": "baseline", **{c: b[c] for c in COLS}})
            for o in OPTS:
                f = FIN / f"metrics_{m}_{t}_{o}.json"
                if f.exists():
                    x = jload(f); rows.append({"model_id": x["model_id"], "task": t, "model": m, "optimizer": o, **{c: x[c] for c in COLS}})
    return pd.DataFrame(rows)


def seed_table():
    rows = []
    for t in TASKS:
        for m in MODELS:
            for o in ("baseline",) + OPTS:
                base = f"{m}_{t}" + ("" if o == "baseline" else f"_{o}")
                ids = [base, base + "_s43", base + "_s44"]
                vals = []
                for i in ids:
                    f = (RES if o == "baseline" else FIN) / f"metrics_{i}.json"
                    if f.exists(): vals.append(jload(f)["macro_f1"])
                if len(vals) == 3:
                    rows.append({"task": t, "model": m, "optimizer": o, "macro_f1_mean": np.mean(vals), "macro_f1_std": np.std(vals, ddof=1), "values": str([round(v, 4) for v in vals])})
    return pd.DataFrame(rows)


def search_table():
    rows = []
    for f in sorted(glob.glob(str(OUT / "runs" / "*.json"))):
        r = jload(f); h = r["history"]; fm = FIN / f"metrics_{r['model']}_{r['task']}_{r['method']}.json"
        x = jload(fm) if fm.exists() else {}
        rows.append({"task": r["task"], "model": r["model"], "optimizer": r["method"], "budget": r["budget"], "n_evals": r["n_evals"], "n_calls": r["n_calls"],
                     "best_fitness": r["best_fitness"], "n_features": int(sum(r["mask"])), "wall_s": r["wall_s"], "cpu_s": r["cpu_s"],
                     "mean_eval_s": float(np.mean([e["eval_s"] for e in h])), "eval_of_best": next(e["eval"] for e in h if e["best_so_far"] == r["best_fitness"]),
                     "val_macro_f1_full": x.get("val_macro_f1_full"), "train_s_full": x.get("train_s"), "hp": str(r["hp"])})
    return pd.DataFrame(rows)


def resource_table():
    base = pd.read_csv(RES / "benchmark_docker.csv"); new = pd.read_csv(OUT / "benchmark_docker.csv")
    d = pd.concat([base[base.model_id.isin([f"{m}_{t}" for m in MODELS for t in TASKS])], new])
    keep = [c for c in ("model_id", "size_kb", "peak_rss_mb", "cpu_util_pct", "lat_median_ms", "lat_mean_ms", "throughput_b1024_per_s", "failed", "oom") if c in d]
    return d[keep]


if __name__ == "__main__":
    metrics_table().to_csv(OUT / "optimization_metrics.csv", index=False)
    seed_table().to_csv(OUT / "optimization_seed_stability.csv", index=False)
    search_table().to_csv(OUT / "search_summary.csv", index=False)
    if (OUT / "benchmark_docker.csv").exists():
        resource_table().to_csv(OUT / "optimization_resources.csv", index=False)
    print("tables written")
