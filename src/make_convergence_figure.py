"""Convergence figure from saved histories only (no search is re-run)."""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS, MODELS, OPTS = ["binary", "multi"], ["dt", "rf", "xgb", "mlp"], ["ga", "pso", "aco", "random"]
BUDGET = {"binary": [800, 800, 800, 600], "multi": [800, 800, 650, 580]}
NAMES = {"dt": "Decision Tree", "rf": "Random Forest", "xgb": "XGBoost", "mlp": "MLP"}
LAB = {"ga": "GA", "pso": "PSO", "aco": "ACO", "random": "Random search"}
STYLE = {"ga": ("#0072B2", "-"), "pso": ("#D55E00", "--"), "aco": ("#009E73", "-."), "random": ("#000000", ":")}

H, ev, issues = {}, {}, []
for t in TASKS:
    for m, b in zip(MODELS, BUDGET[t]):
        for o in OPTS:
            d = json.load(open(f"{ROOT}/results/optimization/runs/{t}_{m}_{o}.json"))
            h = [r["best_so_far"] for r in d["history"]]
            H[t, m, o] = h; ev[t, m, o] = [r["eval"] for r in d["history"]]
            if len(h) != b or d["budget"] != b or ev[t, m, o] != list(range(1, len(h) + 1)):
                issues.append((t, m, o, len(h), b, d["budget"]))
print("Runs with history length != expected budget (or file budget differs / eval not 1..n):")
for i in issues: print("  ", i)
if not issues: print("   none")

fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.4), sharex=False)
plt.rcParams.update({"font.size": 7})
for r, t in enumerate(TASKS):
    for c, m in enumerate(MODELS):
        ax = axs[r, c]
        for o in OPTS:
            h = H[t, m, o]; col, ls = STYLE[o]
            ax.plot(range(1, len(h) + 1), h, color=col, ls=ls, lw=1.0, label=LAB[o])
        ax.axvline(20, color="gray", ls=(0, (1, 2)), lw=0.6)
        ax.set_xlim(0, BUDGET[t][c])
        lo = min(H[t, m, o][19] for o in OPTS); hi = max(H[t, m, o][-1] for o in OPTS)
        ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.08 * (hi - lo))  # zoom past the first-20 rise
        ax.set_title(f"{NAMES[m]}, {'binary' if t == 'binary' else '8-class'}", fontsize=7)
        ax.tick_params(labelsize=6.5)
        if r == 1: ax.set_xlabel("Unique evaluations", fontsize=7)
        if c == 0: ax.set_ylabel("Best search fitness", fontsize=7)
hd, lb = axs[0, 0].get_legend_handles_labels()
fig.legend(hd, lb, loc="upper center", ncol=4, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
    fig.savefig(f"{ROOT}/figures/convergence.{ext}", **kw)
print("wrote figures/convergence.png, figures/convergence.svg")

# check 1: share of final improvement over best of first 20
mine = {"aco": (.62, .85), "ga": (.68, .86), "pso": (.71, .88), "random": (.59, .82)}
print("\nMean share of final improvement over best of first 20 (8 runs each, positive improvement only)")
print("opt     n  | k=100 mine  yours | k=200 mine  yours")
for o in OPTS:
    s100, s200 = [], []
    for t in TASKS:
        for m in MODELS:
            h = H[t, m, o]; imp = h[-1] - h[19]
            if imp > 0: s100.append((h[99] - h[19]) / imp); s200.append((h[199] - h[19]) / imp)
    print(f"{o:7s} {len(s100)} |        {np.mean(s100):.3f}  {mine[o][0]:.2f} |        {np.mean(s200):.3f}  {mine[o][1]:.2f}")

# check 2: per run values and final vs search_summary
ss = pd.read_csv(f"{ROOT}/results/optimization/search_summary.csv").set_index(["task", "model", "optimizer"])
print("\ntask   model opt     len |      @20      @100      @200     final | summary best_fitness | diff")
for t in TASKS:
    for m in MODELS:
        for o in OPTS:
            h = H[t, m, o]; s = ss.loc[(t, m, o), "best_fitness"]
            g = lambda k: f"{h[k-1]:.6f}" if len(h) >= k else "   n/a  "
            print(f"{t:6s} {m:5s} {o:7s} {len(h):3d} | {g(20)} {g(100)} {g(200)} {h[-1]:.6f} | {s:.6f} | {h[-1]-s:+.2e}")

# check 3: oddities
print("\nOddities: flat runs / late last improvement / non-monotone")
for (t, m, o), h in H.items():
    a = np.array(h); last = int(np.argmax(a >= a[-1])) + 1
    nm = bool((np.diff(a) < -1e-12).any())
    if a[-1] == a[19] or nm or last < 0.5 * len(h):
        print(f"  {t} {m} {o}: n={len(h)} final==@20:{a[-1]==a[19]} last_improvement_at={last} nonmonotone={nm}")
