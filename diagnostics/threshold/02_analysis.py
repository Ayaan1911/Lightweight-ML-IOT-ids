"""Validation-chosen operating points for the 4 binary baselines, applied unchanged to test, plus balanced-subset supplement.
Needs scores_*.npz from 01_scores.py. Order enforced here: (2) integrity at 0.5 -> (3) thresholds from VAL ONLY, written to
thresholds.csv -> (4) test evaluation -> (5) supplement. Nothing is chosen using test results."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src import data
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
MODELS = ["dt", "rf", "xgb", "mlp"]; NAME = {"dt": "Decision Tree", "rf": "Random Forest", "xgb": "XGBoost", "mlp": "MLP"}
PUB = {"dt": (0.7854, 0.4211), "rf": (0.8021, 0.4699), "xgb": (0.8194, 0.4023), "mlp": (0.7800, 0.5091)}
Z = {m: np.load(OUT / f"scores_{m}.npz") for m in MODELS}


def metrics(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    f = lambda a, b, c: 0.0 if a == 0 else 2 * a / (2 * a + b + c)
    return {"recall": tp / (tp + fn), "fpr": fp / (fp + tn), "macro_f1": (f(tp, fp, fn) + f(tn, fn, fp)) / 2, "accuracy": (tp + tn) / len(y)}


# ---- step 2: integrity at threshold 0.5 (test scores, no selection)
bad = []
for m in MODELS:
    y, s = Z[m]["test_y"], Z[m]["test_score"]; p = (s > 0.5).astype(int); r = metrics(y, p)
    st = json.load(open(ROOT / "results" / f"metrics_{m}_binary.json")); sp = np.load(ROOT / "results" / f"pred_{m}_binary.npz")
    ok = [r["macro_f1"] == st["macro_f1"], r["fpr"] == st["fpr"], bool((sp["y_pred"] == p).all()),
          round(r["macro_f1"], 4) == PUB[m][0], round(r["fpr"], 4) == PUB[m][1]]
    print(f"integrity {m}: F1 {r['macro_f1']:.4f} FPR {r['fpr']:.4f} | equals metrics json F1/FPR, stored preds, published 4dp: {ok}")
    if not all(ok): bad.append(m)
if bad: sys.exit(f"INTEGRITY CHECK FAILED for {bad}: stopping, nothing else written")


# ---- step 3: thresholds from validation only
def val_choice(y, s):
    o = np.argsort(-s, kind="stable"); ss, yy = s[o], y[o]
    last = np.r_[ss[1:] != ss[:-1], True]
    tp = np.cumsum(yy)[last]; fp = np.cumsum(1 - yy)[last]; t = ss[last]
    P, N = yy.sum(), (1 - yy).sum(); fn, tn = P - tp, N - fp
    f1a = np.where(tp == 0, 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1)); f1b = np.where(tn == 0, 0, 2 * tn / np.maximum(2 * tn + fn + fp, 1))
    mf = (f1a + f1b) / 2; rec = tp / P
    out = {}
    for tgt in (0.98, 0.99):
        out[f"recall>={tgt}"] = t[np.argmax(rec >= tgt)]            # t is descending: first hit = highest cutoff
    out["max_val_macro_f1"] = t[np.argmax(mf)]                      # first max = highest cutoff among ties
    return out


rows = []
for m in MODELS:
    for rule, t in val_choice(Z[m]["val_y"], Z[m]["val_score"]).items():
        rows.append({"model": m, "rule": rule, "threshold": float(t), "pred_attack_if": "score >= threshold"})
thr = pd.DataFrame(rows); thr.to_csv(OUT / "thresholds.csv", index=False)
print("\nthresholds written to thresholds.csv (validation only):\n", thr.to_string(index=False))

# ---- step 4: apply unchanged to test; val values at the same threshold
def boot(y, p, B=1000):
    rng = np.random.default_rng(42); n = len(y); f, r = [], []
    for _ in range(B):
        i = rng.integers(0, n, n); mm = metrics(y[i], p[i]); f.append(mm["fpr"]); r.append(mm["recall"])
    return np.percentile(f, [2.5, 97.5]), np.percentile(r, [2.5, 97.5])


ref = [{"model": m, "rule": "default 0.5 (reference)", "threshold": 0.5, "pred_attack_if": "score > threshold"} for m in MODELS]
res = []
for _, row in pd.concat([thr, pd.DataFrame(ref)], ignore_index=True).iterrows():
    m, t = row["model"], row["threshold"]; ge = row["pred_attack_if"] == "score >= threshold"
    pv = lambda s: (s >= t if ge else s > t).astype(int)
    v = metrics(Z[m]["val_y"], pv(Z[m]["val_score"])); te = metrics(Z[m]["test_y"], pv(Z[m]["test_score"]))
    fc, rc = boot(Z[m]["test_y"], pv(Z[m]["test_score"]))
    res.append({"model": m, "rule": row["rule"], "threshold": t,
                **{f"val_{k}": v[k] for k in v}, **{f"test_{k}": te[k] for k in te},
                "gap_recall": te["recall"] - v["recall"], "gap_fpr": te["fpr"] - v["fpr"],
                "test_fpr_ci_lo": fc[0], "test_fpr_ci_hi": fc[1], "test_recall_ci_lo": rc[0], "test_recall_ci_hi": rc[1]})
R = pd.DataFrame(res); R.to_csv(OUT / "operating_points.csv", index=False)
print("\n", R[["model", "rule", "threshold", "val_recall", "test_recall", "val_fpr", "test_fpr", "val_macro_f1", "test_macro_f1", "test_accuracy"]].round(4).to_string(index=False))

# ---- step 5: balanced-subset supplement (threshold 0.5)
ym = data.load("test", "multi")[1]; yb = Z["dt"]["test_y"]
assert ((ym > 0) == (yb == 1)).all()
ben = np.where(yb == 0)[0]; K = len(ben)
cats = {c: np.where(ym == c)[0] for c in range(1, 8)}; Nat = sum(len(v) for v in cats.values())
q = {c: K * len(v) / Nat for c, v in cats.items()}; k = {c: int(np.floor(v)) for c, v in q.items()}
for c in sorted(q, key=lambda c: q[c] - k[c], reverse=True)[:K - sum(k.values())]: k[c] += 1
assert sum(k.values()) == K
sub = {m: [] for m in MODELS}
for seed in range(1, 101):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([ben] + [rng.choice(cats[c], k[c], replace=False) for c in cats])
    for m in MODELS:
        mm = metrics(yb[idx], (Z[m]["test_score"][idx] > 0.5).astype(int)); sub[m].append((mm["accuracy"], mm["macro_f1"]))
B = pd.DataFrame([{"model": m, "n_subsets": 100, "rows_per_subset": 2 * K, "attack_per_category": json.dumps(k),
                   "accuracy_mean": np.mean([a for a, _ in v]), "accuracy_sd": np.std([a for a, _ in v], ddof=1),
                   "macro_f1_mean": np.mean([f for _, f in v]), "macro_f1_sd": np.std([f for _, f in v], ddof=1)} for m, v in sub.items()])
B.to_csv(OUT / "balanced_supplement.csv", index=False)
print("\nSUPPLEMENT (threshold 0.5, 100 balanced subsets, SD ddof=1)\n", B.drop(columns="attack_per_category").round(4).to_string(index=False))

# ---- figure: test FPR vs recall with validation-chosen points
fig, ax = plt.subplots(figsize=(5.2, 3.6)); col = dict(zip(MODELS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]))
mk = {"recall>=0.98": "o", "recall>=0.99": "s", "max_val_macro_f1": "^", "default 0.5 (reference)": "x"}
for m in MODELS:
    y, s = Z[m]["test_y"], Z[m]["test_score"]; o = np.argsort(-s, kind="stable"); ss = s[o]; last = np.r_[ss[1:] != ss[:-1], True]
    ax.plot((np.cumsum(1 - y[o]) / (1 - y).sum())[last], (np.cumsum(y[o]) / y.sum())[last], color=col[m], lw=1, label=NAME[m])
    for _, r in R[R.model == m].iterrows():
        ax.plot(r.test_fpr, r.test_recall, mk[r.rule], color=col[m], mec="k" if r.rule != "default 0.5 (reference)" else col[m], ms=5)
ax.set_xlabel("False positive rate (benign flagged as attack), test"); ax.set_ylabel("Attack recall, test"); ax.set_ylim(0.9, 1.003)
h = [plt.Line2D([], [], color="gray", marker=v, ls="", label=k) for k, v in mk.items()]
l1 = ax.legend(loc="lower right", fontsize=7); ax.add_artist(l1); ax.legend(handles=h, loc="center right", fontsize=7, title="validation-chosen", title_fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fpr_vs_recall_test.png", dpi=200)
