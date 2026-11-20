"""Validation-only diagnostic: why is the binary FPR high?  Read-only w.r.t. models/, results/, data_processed/.
Run from project root:  .venv\\Scripts\\python diagnostics\\run_diagnostics.py   Outputs go to diagnostics/ only.
NEVER touches the test split (no data.load('test'), no score_test, no test_access.log writes)."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
OUT = ROOT / "diagnostics"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_sample_weight
from src.common import LABEL2CAT, CLASSES, PROC, MOD, RES, SEED, jload, norm_label
from src import data, predictors

SAMPLE = PROC / "sample.parquet"
pd.set_option("display.width", 220)


def hashes(X):
    return pd.util.hash_pandas_object(X, index=False).to_numpy()


# ---------------------------------------------------------------- step 1
def step1():
    df = pd.read_parquet(SAMPLE)
    feat = [c for c in df.columns if c != "Label"]
    steps, log = [], []

    def snap(name, d):
        cat = d["Label"].map(LABEL2CAT)
        s = d["Label"].value_counts().rename(name)
        steps.append(s)
        ben = int((cat == "Benign").sum()); att = int(len(d) - ben)
        log.append({"step": name, "rows": len(d), "benign": ben, "attack": att, "benign_pct": 100 * ben / len(d)})

    snap("0 sampled", df)
    df = df[df["Label"] != "NAN"].reset_index(drop=True); snap("1 drop missing label", df)
    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan); df = df.dropna().reset_index(drop=True); snap("2 drop NaN/inf rows", df)
    df["cat"] = df["Label"].map(LABEL2CAT)
    df = df.drop_duplicates(subset=feat + ["Label"]).reset_index(drop=True); snap("3 exact duplicates (features+label)", df)
    h = hashes(df[feat])
    nun = df.groupby(h)["cat"].transform("nunique")
    conf = df[nun > 1]
    combos = conf.groupby(h[nun > 1])["cat"].apply(lambda x: " + ".join(sorted(set(x)))).value_counts()
    combos.rename("n_vectors").to_csv(OUT / "step1_conflict_category_combos.csv")
    conf["cat"].value_counts().rename("rows_removed").to_csv(OUT / "step1_conflict_rows_by_category.csv")
    conf["Label"].value_counts().rename("rows_removed").to_csv(OUT / "step1_conflict_rows_by_label.csv")
    df = df[nun == 1].reset_index(drop=True); snap("4 conflicting task labels", df)
    df = df.drop_duplicates(subset=feat).reset_index(drop=True); snap("5 identical vectors, same category", df)
    pd.DataFrame(log).to_csv(OUT / "step1_cleaning_balance.csv", index=False)
    pd.concat(steps, axis=1).fillna(0).astype(int).to_csv(OUT / "step1_rows_per_label_per_step.csv")
    # benign vs attack removed per step
    L = pd.DataFrame(log); L["benign_removed"] = -L.benign.diff().fillna(0).astype(int); L["attack_removed"] = -L.attack.diff().fillna(0).astype(int)
    L.to_csv(OUT / "step1_cleaning_balance.csv", index=False)
    return L, combos


# ---------------------------------------------------------------- step 2
def step2():
    Xv, yv = data.load("val", "binary")
    Xn = Xv.to_numpy(np.float32)
    rows = []
    for n in ("dt", "rf", "xgb", "mlp"):
        mid = f"{n}_binary"
        p = predictors.load(mid)(Xn)
        pr, rc, f1, sup = precision_recall_fscore_support(yv, p, labels=[0, 1], zero_division=0)
        cm = confusion_matrix(yv, p, labels=[0, 1])
        pd.DataFrame(cm, index=["Benign", "Attack"], columns=["pred Benign", "pred Attack"]).to_csv(OUT / f"step2_cm_val_{mid}.csv")
        rows.append({"model": mid, "benign_precision": pr[0], "benign_recall": rc[0], "benign_f1": f1[0], "attack_precision": pr[1],
                     "attack_recall": rc[1], "attack_f1": f1[1], "macro_f1": f1.mean(), "fpr(benign->attack)": cm[0, 1] / cm[0].sum(),
                     "val_benign": int(sup[0]), "val_attack": int(sup[1])})
    t = pd.DataFrame(rows); t.to_csv(OUT / "step2_val_benign_vs_attack.csv", index=False)
    return t


# ---------------------------------------------------------------- step 3
def step3():
    Xt, yt = data.load("train", "binary"); Xv, yv = data.load("val", "binary")
    out = {}
    ben_v, att_t, ben_t = Xv[yv == 0], Xt[yt == 1], Xt[yt == 0]
    out["val_benign_rows"] = len(ben_v); out["train_attack_rows"] = len(att_t); out["train_benign_rows"] = len(ben_t)
    for dec in (None, 4, 2, 0):
        a = att_t if dec is None else att_t.round(dec); b = ben_v if dec is None else ben_v.round(dec)
        hit = np.isin(hashes(b), hashes(a))
        out[f"val_benign_with_identical_train_attack_vector(round={dec})"] = int(hit.sum())
        out[f"  share(round={dec})"] = float(hit.mean())
    # nearest-neighbour overlap in standardised space
    z = np.load(MOD / "scaler.npz"); mu, sd = z["mean"].astype(np.float32), z["scale"].astype(np.float32)
    S = lambda X: ((X.to_numpy(np.float32) - mu) / sd).astype(np.float32)
    rng = np.random.default_rng(SEED)
    att_v = Xv[yv == 1]; att_v = att_v.iloc[rng.choice(len(att_v), len(ben_v), replace=False)]
    nb = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(S(ben_t))
    na = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(S(att_t))
    allt = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(S(Xt))
    res = {}
    for name, q in (("val_benign", ben_v), ("val_attack(sample, same n)", att_v)):
        Q = S(q); db, _ = nb.kneighbors(Q, 1); da, _ = na.kneighbors(Q, 1)
        _, ix = allt.kneighbors(Q, 5); lab = yt[ix]
        res[name] = {"n": len(q), "median_dist_to_nearest_train_benign": float(np.median(db)), "median_dist_to_nearest_train_attack": float(np.median(da)),
                     "share_nearest_attack_closer_than_nearest_benign": float((da[:, 0] < db[:, 0]).mean()),
                     "share_1NN_is_attack": float(lab[:, 0].mean()), "mean_share_of_5NN_that_are_attack": float(lab.mean()),
                     "share_all_5NN_attack": float((lab.mean(1) == 1).mean()), "share_all_5NN_benign": float((lab.mean(1) == 0).mean())}
    out["train_attack_share"] = float(yt.mean()); out["nn"] = res
    jdump = json.dumps(out, indent=2); (OUT / "step3_separability.json").write_text(jdump)
    return out


# ---------------------------------------------------------------- steps 4/5
def fit_sweep(Xtr, ytr, Xva, yva, weighted=True):
    m = xgb.XGBClassifier(tree_method="hist", n_jobs=-1, random_state=SEED, n_estimators=300, max_depth=10, learning_rate=0.1)
    m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr) if weighted else None)
    return m.predict_proba(Xva)[:, 1]


def sweep(p, y, tag):
    fpr, tpr, thr = roc_curve(y, p)
    thr_grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows = []
    for t in thr_grid:
        pr = (p >= t).astype(int); cm = confusion_matrix(y, pr, labels=[0, 1])
        rows.append({"threshold": t, "fpr": cm[0, 1] / cm[0].sum(), "attack_recall": cm[1, 1] / cm[1].sum(), "macro_f1": f1_score(y, pr, average="macro")})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / f"{tag}_threshold_sweep.csv", index=False)
    lv = []
    for r in (0.80, 0.90, 0.95, 0.98, 0.99, 0.995):
        i = np.searchsorted(tpr, r); lv.append({"target_attack_recall": r, "fpr_at_that_recall": float(fpr[min(i, len(fpr) - 1)]), "threshold": float(thr[min(i, len(thr) - 1)])})
    lv = pd.DataFrame(lv); lv.to_csv(OUT / f"{tag}_fpr_at_recall.csv", index=False)
    summ = {"auc_roc": float(roc_auc_score(y, p)), "avg_precision_benign_as_positive": float(average_precision_score(1 - y, 1 - p)),
            "best_val_macro_f1_over_thresholds": float(tab.macro_f1.max()), "threshold_of_best": float(tab.threshold[tab.macro_f1.idxmax()]),
            "fpr_at_0.5": float(tab.set_index("threshold").loc[0.5, "fpr"]), "recall_at_0.5": float(tab.set_index("threshold").loc[0.5, "attack_recall"])}
    return fpr, tpr, tab, lv, summ


def variant_data():
    df = pd.read_parquet(SAMPLE)
    feat = [c for c in df.columns if c != "Label"]
    df = df[df["Label"] != "NAN"]
    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan); df = df.dropna()
    n_before = len(df)
    df = df.drop_duplicates(subset=feat + ["Label"]).reset_index(drop=True)  # ONLY exact same-label duplicates
    df["y_multi"] = df["Label"].map(LABEL2CAT); df["y_binary"] = (df["y_multi"] != "Benign").astype(int)
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=df.y_multi, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=df.y_multi.iloc[rest], random_state=SEED)  # te is NOT used anywhere
    h = hashes(df[feat])
    info = {"rows_after_exact_dedup": len(df), "removed_exact_dups": n_before - len(df), "benign": int((df.y_binary == 0).sum()),
            "attack": int((df.y_binary == 1).sum()), "benign_pct": float(100 * (df.y_binary == 0).mean()),
            "val_rows_whose_vector_is_in_train": float(np.isin(h[va], h[tr]).mean()),
            "val_benign_rows_whose_vector_is_in_train": float(np.isin(h[va][df.y_binary.iloc[va].to_numpy() == 0], h[tr]).mean()),
            "val_benign_vector_in_train_with_ATTACK_label": float(np.isin(h[va][df.y_binary.iloc[va].to_numpy() == 0], h[tr][df.y_binary.iloc[tr].to_numpy() == 1]).mean()),
            "train/val sizes": [len(tr), len(va)]}
    return df, feat, tr, va, info


def plot(curves, path, title):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for lab, (fpr, tpr, ls) in curves.items():
        ax.plot(fpr, tpr, ls, label=lab)
    ax.set_xlabel("false positive rate (benign flagged as attack)"); ax.set_ylabel("attack recall"); ax.set_title(title, fontsize=9)
    ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def steps45():
    Xt, yt = data.load("train", "binary"); Xv, yv = data.load("val", "binary")
    out, curves = {}, {}
    for w in (True, False):
        tag = "step4_" + ("balanced" if w else "unweighted")
        p = fit_sweep(Xt.to_numpy(np.float32), yt, Xv.to_numpy(np.float32), yv, w)
        fpr, tpr, tab, lv, s = sweep(p, yv, tag); out[tag] = s; curves[f"cleaned data, {'balanced' if w else 'unweighted'} XGB"] = (fpr, tpr, "-" if w else "--")
    df, feat, tr, va, info = variant_data(); out["variant_info"] = info
    X = df[feat].to_numpy(np.float32); y = df.y_binary.to_numpy()
    p = fit_sweep(X[tr], y[tr], X[va], y[va], True)
    fpr, tpr, tab, lv, s = sweep(p, y[va], "step5_balanced_exactdedup_only"); out["step5_balanced_exactdedup_only"] = s
    curves["exact-dup-only data, balanced XGB (own split)"] = (fpr, tpr, ":")
    plot(curves, OUT / "fpr_vs_recall.png", "Validation only: FPR vs attack recall (threshold sweep)")
    np.savez(OUT / "step5_split_indices.npz", train=tr, val=va)  # val/train only; test indices deliberately not saved
    (OUT / "step4_5_summary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    L, combos = step1(); print(L.round(2).to_string()); print(combos.head(12))
    t2 = step2(); print(t2.round(4).to_string())
    s3 = step3(); print(json.dumps(s3, indent=1))
    s45 = steps45(); print(json.dumps(s45, indent=1))
