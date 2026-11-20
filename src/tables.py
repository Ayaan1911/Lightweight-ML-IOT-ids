"""Aggregate metrics + benchmarks into comparison tables and plots. python -m src.tables"""
import glob, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import *

RES_COLS = ["size_kb", "peak_rss_mb", "peak_rss_increase_mb", "cpu_util_pct", "lat_mean_ms", "lat_median_ms", "lat_p95_ms", "lat_p99_ms",
            "throughput_b1024_per_s"]
DET = ["macro_f1", "accuracy", "precision", "recall", "fpr"]


def metrics_df():
    return pd.DataFrame([jload(f) for f in sorted(glob.glob(str(RES / "metrics_*.json")))]).set_index("model_id")


def bench_df(tag):
    p = RES / f"benchmark_{tag}.csv"
    return pd.read_csv(p).set_index("model_id") if p.exists() else pd.DataFrame()


def family(mid):
    return mid.split("_")[0]


def baseline_table(tag="native"):
    m, b = metrics_df(), bench_df(tag)
    rows = []
    for name, label in (("dt", "Decision Tree"), ("rf", "Random Forest"), ("xgb", "XGBoost"), ("mlp", "MLP")):
        r = {"model": label}
        for task in ("binary", "multi"):
            mid = f"{name}_{task}"
            for c in DET:
                r[f"{task}_{c}"] = m.loc[mid, c] if mid in m.index else np.nan
        for c in RES_COLS:
            r[c] = b.loc[f"{name}_binary", c] if f"{name}_binary" in b.index and c in b else np.nan
            r[f"multi_{c}"] = b.loc[f"{name}_multi", c] if f"{name}_multi" in b.index and c in b else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(RES / f"baseline_comparison_{tag}.csv", index=False)
    return t


def reduced_table(tag="native"):
    """Every variant with change vs its original (original = baseline of same family/task)."""
    m, b = metrics_df(), bench_df(tag)
    rows = []
    for mid in m.index:
        parts = mid.split("_")
        fam, task, var = parts[0], parts[1], "_".join(parts[2:]) or "baseline"
        if mid.endswith(tuple(f"_s{s}" for s in (43, 44))):
            continue
        orig = f"{fam}_{task}"
        r = {"model_id": mid, "family": fam, "task": task, "variant": var, "macro_f1": m.loc[mid, "macro_f1"]}
        if orig in m.index:
            r["d_macro_f1"] = r["macro_f1"] - m.loc[orig, "macro_f1"]
        for c in RES_COLS:
            if mid in b.index and c in b:
                r[c] = b.loc[mid, c]
                if orig in b.index and b.loc[orig, c]:
                    r[f"{c}_ratio"] = b.loc[mid, c] / b.loc[orig, c]
        if mid in b.index and "throughput_b1024_per_s" in b:
            r["throughput_ratio"] = r.pop("throughput_b1024_per_s_ratio", np.nan)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(RES / f"reduced_vs_original_{tag}.csv", index=False)
    return t


def scatter(t, task, tag="native"):
    for xcol, xl, fn in (("size_kb", "model size (KB, log)", "size"), ("lat_median_ms", "batch-1 median latency (ms, log)", "latency")):
        fig, ax = plt.subplots(figsize=(7, 5))
        d = t[(t.task == task) & t[xcol].notna()]
        for fam, mk in zip(("dt", "rf", "xgb", "mlp"), "osD^"):
            s = d[d.family == fam]
            ax.scatter(s[xcol], s.macro_f1, marker=mk, label=fam, s=40)
            for _, r in s.iterrows():
                ax.annotate(r.variant if r.variant != "baseline" else "base", (r[xcol], r.macro_f1), fontsize=5)
        ax.set_xscale("log"); ax.set_xlabel(xl); ax.set_ylabel(f"test macro F1 ({task})"); ax.legend(); ax.grid(alpha=.3)
        ax.set_title(f"Macro F1 vs {fn} - {task} ({tag})")
        fig.tight_layout(); fig.savefig(FIG / f"pareto_{tag}_{task}_{fn}.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    for tag in ("docker", "native"):
        print(tag); print(baseline_table(tag).T.round(4))
        t = reduced_table(tag)
        for task in ("binary", "multi"):
            scatter(t, task, tag)
