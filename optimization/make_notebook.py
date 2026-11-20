"""Build iot_ids_optimization.ipynb (+ the pointer section in iot_ids.ipynb) from the saved CSVs.
python -m optimization.make_notebook ; then nbconvert --execute both notebooks."""
import json, numpy as np, pandas as pd, nbformat as nbf
from src.common import ROOT, jload
from optimization import tables as T

O = ROOT / "results" / "optimization"
M = pd.read_csv(O / "optimization_metrics.csv"); S = pd.read_csv(O / "optimization_seed_stability.csv"); R = pd.read_csv(O / "search_summary.csv")


def what_happened():
    """Plain statements generated only from the CSVs; a result inside 2 baseline seed-SDs is called noise."""
    L = []
    for t in T.TASKS:
        for m in T.MODELS:
            g = M[(M.task == t) & (M.model == m)].set_index("optimizer"); b = g.loc["baseline"]
            opt = g.drop("baseline").drop("random", errors="ignore")
            bo = opt.macro_f1.idxmax(); d = opt.macro_f1[bo] - b.macro_f1; rnd = g.macro_f1["random"]
            ss = S[(S.task == t) & (S.model == m)].set_index("optimizer").macro_f1_std
            sd = float(max(ss["baseline"], ss[bo]))
            if d <= 0: v = "no optimizer beat the baseline"
            elif d < 2 * sd: v = f"the gain is within 2x the larger seed SD of baseline and winner ({sd:.4f}), so it is not distinguishable from seed noise"
            else: v = f"the gain is larger than 2x the larger seed SD ({sd:.4f}); note seed SD covers training randomness only, not split-to-split variation"
            vr = "beat" if opt.macro_f1[bo] > rnd else "did not beat"
            vn = " (inside 2x seed SD, so not a reliable difference)" if abs(opt.macro_f1[bo] - rnd) < 2 * sd else ""
            L.append(f"- **{t} {m}**: baseline macro F1 {b.macro_f1:.4f}; best optimizer {bo.upper()} {opt.macro_f1[bo]:.4f} ({d:+.4f}); {v}. "
                     f"It {vr} random search ({rnd:.4f}) by {opt.macro_f1[bo]-rnd:+.4f}{vn}. FPR baseline {b.fpr:.4f} vs {bo.upper()} {opt.fpr[bo]:.4f}. "
                     f"GA/PSO/ACO that did not beat the baseline: {', '.join(o.upper() for o in opt.index if opt.macro_f1[o] <= b.macro_f1) or 'none'}.")
    return L


GLANCE = '''import pandas as pd, numpy as np, json, matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Markdown
O = Path("results/optimization")
M = pd.read_csv(O / "optimization_metrics.csv")
cols = ["macro_f1", "accuracy", "precision", "recall", "fpr"]
OPT = {"baseline": 0, "ga": 1, "pso": 2, "aco": 3, "random": 4}
M["o"] = M.optimizer.map(OPT)
M = M.sort_values(["task", "model", "o"]).reset_index(drop=True)
b = M[M.optimizer == "baseline"].set_index(["model", "task"])[cols]
for c in cols:
    M["d_" + c] = M[c].values - [b.loc[(m, t), c] for m, t in zip(M.model, M.task)]
T = M[["model_id", "task", "optimizer"] + cols + ["d_" + c for c in cols]]

def css(df):
    out = pd.DataFrame("", index=df.index, columns=df.columns)
    for (m, t), g in M.groupby(["model", "task"]):
        for c in cols:
            best = g[c].min() if c == "fpr" else g[c].max()
            out.loc[g.index[g[c] == best], c] = "background-color:#b7e1b7;font-weight:bold"
            for i in g.index:
                d = M.loc[i, "d_" + c]
                if M.loc[i, "optimizer"] == "baseline" or d == 0: continue
                good = d < 0 if c == "fpr" else d > 0
                out.loc[i, "d_" + c] = "background-color:%s" % ("#b7e1b7" if good else "#f4b6b6")
    return out

sty = (T.style.apply(css, axis=None).format({**{c: "{:.4f}" for c in cols}, **{"d_" + c: "{:+.4f}" for c in cols}}).hide(axis="index")
       .set_caption("Test-set metrics, single final model (seed 42). Green bold = best per model/task; delta columns vs baseline: green better, red worse (lower FPR is better)."))
display(sty)'''

BARS = '''fig, ax = plt.subplots(2, 2, figsize=(13, 7))
names = {"dt": "DT", "rf": "RF", "xgb": "XGB", "mlp": "MLP"}
for r, t in enumerate(["binary", "multi"]):
    for k, (c, lab) in enumerate([("macro_f1", "macro F1 (higher better)"), ("fpr", "FPR (lower better)")]):
        p = M[M.task == t].pivot(index="model", columns="optimizer", values=c).loc[["dt", "rf", "xgb", "mlp"], list(OPT)]
        p.rename(index=names).plot.bar(ax=ax[r, k], rot=0, width=0.8)
        ax[r, k].set_title(f"{t}: {lab}"); ax[r, k].set_xlabel(""); ax[r, k].legend(fontsize=7, ncol=5)
        if c == "macro_f1": ax[r, k].set_ylim(max(0, p.values.min() - 0.05), min(1, p.values.max() + 0.03))
plt.tight_layout(); plt.show()'''

METHOD_MD = '''## Method
**Encoding (shared by GA, PSO, ACO and random search).** One vector in [0,1]^(39+h): the first 39 entries are feature switches (kept if > 0.5; if fewer than 5 are kept the 5 largest entries are used), followed by *h* hyperparameter entries mapped to the ranges below (integers rounded, `log` ranges mapped on a log scale, categories by equal-width bins).

**Fitness.** macro F1 on a fixed stratified 50,000-row *validation* subsample, for a model trained on a fixed stratified 100,000-row *training* subsample (rare classes keep at least 200 rows), minus 0.001 x (fraction of features used). The MLP uses the training-fitted scaler and its early stopping looks at a 10% hold-out of the training subsample. The optimizer code never reads the held-out test split; the final winners are retrained on the full training split and scored once on test.

**Optimizers.** Same population / swarm / ant count (20), same seed, same first 20 random candidates and the same budget of *unique* evaluations per model and task. GA: tournament(3), BLX-0.5 crossover, Gaussian mutation, 2 elites. PSO: inertia 0.72, c1 = c2 = 1.49. ACO: ACO for continuous domains (ACOR: solution archive of 20, rank weights, Gaussian kernel sampling), an adaptation of ACO to continuous variables, not the classical discrete one. Random search: uniform draws with the same budget.'''

METHOD = '''import sys; sys.path.insert(0, ".")
from optimization import core
rows = [{"model": m, "parameter": s[0], "type": s[1], "range / choices": (f"{s[2]} .. {s[3]}" if s[1] != "cat" else str(s[2]))} for m, v in core.HP.items() for s in v]
display(pd.DataFrame(rows))
R = pd.read_csv(O / "search_summary.csv")
bud = R.groupby(["task", "model"]).agg(budget=("budget", "first"), evals_per_method=("n_evals", "unique"), mean_eval_s=("mean_eval_s", "mean")).reset_index()
bud["evals_per_method"] = bud.evals_per_method.map(lambda v: ", ".join(map(str, v)))
display(bud.style.format({"mean_eval_s": "{:.2f}"}).hide(axis="index").set_caption("Evaluation budget (identical for GA, PSO, ACO, random)"))'''

CONV = '''fig, ax = plt.subplots(2, 4, figsize=(18, 7))
for r, t in enumerate(["binary", "multi"]):
    for k, m in enumerate(["dt", "rf", "xgb", "mlp"]):
        for o in ["ga", "pso", "aco", "random"]:
            h = json.load(open(O / "runs" / f"{t}_{m}_{o}.json"))["history"]
            ax[r, k].plot([e["eval"] for e in h], [e["best_so_far"] for e in h], label=o.upper())
        ax[r, k].set_title(f"{t} {m}"); ax[r, k].set_xlabel("evaluation"); ax[r, k].set_ylabel("best fitness so far")
        # zoom the y axis on the last 80% of the curve so differences are visible
        lo = min(json.load(open(O / "runs" / f"{t}_{m}_{o}.json"))["history"][19]["best_so_far"] for o in ["ga", "pso", "aco", "random"])
        ax[r, k].set_ylim(lo - 0.005, None)
        ax[r, k].legend(fontsize=7)
plt.tight_layout(); plt.show()'''

WIN = '''import pyarrow.parquet as pq
names = [n for n in pq.read_schema("data_processed/clean.parquet").names if not n.startswith("y_")]
rows = []
for o in ["ga", "pso", "aco", "random"]:
    for t in ["binary", "multi"]:
        for m in ["dt", "rf", "xgb", "mlp"]:
            r = json.load(open(O / "runs" / f"{t}_{m}_{o}.json"))
            rows.append({"task": t, "model": m, "optimizer": o, "fitness": r["best_fitness"], "n_features": sum(r["mask"]),
                         "features": ", ".join(n for n, k in zip(names, r["mask"]) if k), "hyperparameters": str(r["hp"])})
W = pd.DataFrame(rows).sort_values(["task", "model", "optimizer"])
pd.set_option("display.max_colwidth", 400)
display(W.style.format({"fitness": "{:.4f}"}).hide(axis="index"))
freq = pd.DataFrame({t: pd.Series(np.sum([json.load(open(O / "runs" / f"{t}_{m}_{o}.json"))["mask"] for m in ["dt","rf","xgb","mlp"] for o in ["ga","pso","aco","random"]], axis=0), index=names) for t in ["binary", "multi"]})
display(freq.sort_values("multi", ascending=False).head(12).style.set_caption("How often a feature is kept across the 16 winners per task (max 16)"))'''

RES = '''B = pd.read_csv("results/benchmark_docker.csv").set_index("model_id")
N = pd.read_csv(O / "benchmark_docker.csv").set_index("model_id")
rows = []
for _, r in M.iterrows():
    mid, base = r.model_id, f"{r.model}_{r.task}"
    src = B if r.optimizer == "baseline" else N
    x = src.loc[mid] if mid in src.index else None
    oom = x is None or str(x.get("oom")) == "True" or str(x.get("failed")) == "True"
    bx = B.loc[base]
    rows.append({"model_id": mid, "optimizer": r.optimizer, "size_kb": np.nan if oom else x.size_kb, "peak_rss_mb": np.nan if oom else x.peak_rss_mb,
                 "lat_median_ms": np.nan if oom else x.lat_median_ms, "thr_b1024_per_s": np.nan if oom else x.throughput_b1024_per_s,
                 "status": "OOM/failed at 1 GB" if oom else "ok"})
Rs = pd.DataFrame(rows)
Rs["size_x_baseline"] = Rs.size_kb / Rs.model_id.map(lambda i: Rs.loc[Rs.model_id == i.split("_")[0] + "_" + i.split("_")[1], "size_kb"].iloc[0])
Rs["lat_x_baseline"] = Rs.lat_median_ms / Rs.model_id.map(lambda i: Rs.loc[Rs.model_id == i.split("_")[0] + "_" + i.split("_")[1], "lat_median_ms"].iloc[0])
display(Rs.style.format({"size_kb": "{:.1f}", "peak_rss_mb": "{:.1f}", "lat_median_ms": "{:.4f}", "thr_b1024_per_s": "{:,.0f}", "size_x_baseline": "{:.2f}", "lat_x_baseline": "{:.2f}"}).hide(axis="index")
        .set_caption("Docker --cpus=1 --memory=1g. Baseline rows come from the original benchmark (results/benchmark_docker.csv). A CPU utilisation column is omitted: it is ~100% for every model (see REPORT.md)."))
R = pd.read_csv(O / "search_summary.csv")
cost = R.groupby(["task", "model", "optimizer"]).agg(evals=("n_evals", "first"), cpu_s=("cpu_s", "first"), wall_s=("wall_s", "first"), mean_eval_s=("mean_eval_s", "first"), best_found_at_eval=("eval_of_best", "first")).reset_index()
display(cost.style.format({"cpu_s": "{:.0f}", "wall_s": "{:.0f}", "mean_eval_s": "{:.2f}"}).hide(axis="index").set_caption("Optimization cost per run (4 runs shared the CPU, so wall time includes contention; cpu_s is the process CPU time)"))
print("total optimization CPU hours:", round(R.cpu_s.sum() / 3600, 1), " max wall per combo (h):", round(R.groupby(["task", "model"]).wall_s.max().sum() / 3600, 1))'''

SEEDS = '''S = pd.read_csv(O / "optimization_seed_stability.csv")
display(S.style.format({"macro_f1_mean": "{:.4f}", "macro_f1_std": "{:.4f}"}).hide(axis="index").set_caption("Final configuration retrained with seeds 42, 43, 44 (split fixed); mean and sample SD of test macro F1"))'''


def build():
    nb = nbf.v4.new_notebook(); c = nb.cells
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    c += [md("# IoT intrusion detection: metaheuristic optimization (GA, PSO, ACO)\nPhase 2 of the CICIoT2023 project. Each of the four models is tuned (features and hyperparameters together) by GA, PSO, ACO and a random-search control with an identical budget, and compared with the baselines of `iot_ids.ipynb`.\n\n## Results at a glance"),
          code(GLANCE), code(BARS), md(METHOD_MD), code(METHOD),
          md("## Convergence\nBest fitness so far (validation-subsample macro F1 minus the 0.001 feature penalty) against the number of unique evaluations; y axis starts just below the best value after the first 20 shared random candidates."),
          code(CONV),
          md("## Selected features and hyperparameters of each winner"), code(WIN),
          md("## Resources before and after, and cost of the search"), code(RES),
          md("## Seed noise"), code(SEEDS),
          md("## What happened\nGenerated from `optimization_metrics.csv` and `optimization_seed_stability.csv`. A gain smaller than two seed standard deviations (the larger of baseline and winner) is called noise; seeds change training randomness only, the data split is fixed, so true uncertainty is larger. Selection used validation-subsample fitness only, so a winner can lose on test.\n\n" + "\n".join(what_happened()))]
    nbf.write(nb, ROOT / "iot_ids_optimization.ipynb")
    # pointer section in the finished notebook (append only; existing cells untouched)
    p = ROOT / "iot_ids.ipynb"; old = nbf.read(p, as_version=4)
    if not any("iot_ids_optimization" in "".join(x["source"]) for x in old.cells):
        old.cells.append(md("## Next phase: metaheuristic optimization\nGA, PSO and ACO (plus a random-search control) were applied to all four models; method, results, resources and an honest discussion are in [`iot_ids_optimization.ipynb`](iot_ids_optimization.ipynb) and in the section *Optimization with GA, PSO and ACO* of `REPORT.md`. Nothing above this cell was changed."))
        nbf.write(old, p)


if __name__ == "__main__":
    build()
