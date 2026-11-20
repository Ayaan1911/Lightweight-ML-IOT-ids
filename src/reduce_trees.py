"""Tree reduction grid + variant selection.  python -m src.reduce_trees grid|select"""
import sys, io, time, joblib, numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
from .common import *
from . import data, predictors
from .evaluation import val_f1, score_test

KS = ["all", 30, 20, 15, 10, 5]
DEPTH = {"dt": [None, 20, 15, 10, 8, 5], "rf": [None, 12, 8, 5], "xgb": [20, 12, 8, 5]}
NEST = [100, 50, 25, 10]
GRID_CSV = RES / "tree_reduction_grid.csv"
_C = {}


def arrays(task):
    if task not in _C:
        Xtr, ytr = data.load("train", task); Xva, yva = data.load("val", task)
        _C[task] = (Xtr.to_numpy(np.float32), ytr, Xva.to_numpy(np.float32), yva)
    return _C[task]


def rankings(name, task):
    """Feature order (best first) by mutual information (100k train subsample) and by baseline-model importance."""
    p = RES / f"rankings_{name}_{task}.json"
    if p.exists():
        return jload(p)
    Xtr, ytr, _, _ = arrays(task)
    rng = np.random.default_rng(SEED); ix = rng.choice(len(Xtr), 100_000, replace=False)
    mi = mutual_info_classif(Xtr[ix], ytr[ix], random_state=SEED)
    mid = f"{name}_{task}"
    if name == "xgb":
        b = xgb.Booster(); b.load_model(MOD / f"{mid}.json")
        g = b.get_score(importance_type="gain"); imp = np.array([g.get(f"f{i}", 0.0) for i in range(Xtr.shape[1])])
    else:
        imp = joblib.load(MOD / f"{mid}.joblib").feature_importances_
    out = {"mi": np.argsort(-mi).tolist(), "imp": np.argsort(-imp).tolist(), "mi_score": mi.tolist(), "imp_score": imp.tolist()}
    jdump(out, p)
    return out


def base_choice(name, task):
    return jload(RES / "baseline_config.json")[f"{name}_{task}"]["chosen"]


def fit(name, task, k, rank, depth, nest, seed=SEED):
    Xtr, ytr, _, _ = arrays(task)
    order = rankings(name, task)[rank]
    cols = sorted(order if k == "all" else order[:k])
    X = Xtr[:, cols]
    b = base_choice(name, task)
    if name == "dt":
        m = DecisionTreeClassifier(random_state=seed, max_depth=depth, class_weight=b["class_weight"]).fit(X, ytr)
    elif name == "rf":
        m = RandomForestClassifier(random_state=seed, n_jobs=-1, n_estimators=nest, max_depth=depth, class_weight=b["class_weight"]).fit(X, ytr)
    else:
        m = xgb.XGBClassifier(tree_method="hist", n_jobs=-1, random_state=seed, n_estimators=nest, max_depth=depth,
                              learning_rate=b["learning_rate"])
        m.fit(X, ytr, sample_weight=compute_sample_weight("balanced", ytr) if b["weighted"] else None)
    return m, cols


def size_and_pred(name, m):
    if name == "xgb":
        return len(m.get_booster().save_raw(raw_format="json")) / 1024
    buf = io.BytesIO(); joblib.dump(m, buf); return buf.tell() / 1024


def latency_ms(name, m, Xs, n=300):
    if hasattr(m, "n_jobs"):
        m.n_jobs = 1
    if name == "xgb":
        b = m.get_booster(); b.set_param({"nthread": 1}); f = b.inplace_predict
    else:
        f = m.predict
    for i in range(30):
        f(Xs[i:i + 1])
    t = []
    for i in range(n):
        s = time.perf_counter(); f(Xs[i:i + 1]); t.append(time.perf_counter() - s)
    return float(np.median(t) * 1000)


def grid():
    done = pd.read_csv(GRID_CSV) if GRID_CSV.exists() else pd.DataFrame()
    seen = set(map(tuple, done[["model", "task", "k", "rank", "depth", "n_est"]].astype(str).values)) if len(done) else set()
    for task in ("binary", "multi"):
        _, _, Xva, yva = arrays(task)
        for name in ("dt", "rf", "xgb"):
            rk = ["mi", "imp"]
            combos = []
            if name == "dt":
                combos = [(k, r, d, 0) for r in rk for k in KS for d in DEPTH[name]]
            else:
                bd = DEPTH[name][0]
                combos = [(k, r, bd, 100) for r in rk for k in KS]  # ranking ablation at full-size params
            for phase in (1, 2):
                if phase == 2 and name != "dt":
                    # pick ranking with best mean val F1 in the ablation, then full grid with it
                    d = pd.read_csv(GRID_CSV); d = d[(d.model == name) & (d.task == task)]
                    best = d.groupby("rank").val_macro_f1.mean().idxmax()
                    combos = [(k, best, dd, ne) for k in KS for dd in DEPTH[name] for ne in NEST]
                elif phase == 2:
                    break
                for k, r, d, ne in combos:
                    key = (name, task, str(k), r, str(d), str(ne))
                    if key in seen:
                        continue
                    t0 = time.time()
                    m, cols = fit(name, task, k, r, d, ne)
                    Xs = Xva[:, cols]
                    pv = m.predict(Xs)
                    row = dict(model=name, task=task, k=k, rank=r, depth=d, n_est=ne, n_features=len(cols), cols=str(cols),
                               val_macro_f1=val_f1(yva, pv), size_kb=size_and_pred(name, m), lat_ms=latency_ms(name, m, Xva[:, cols]),
                               train_s=time.time() - t0)
                    pd.DataFrame([row]).to_csv(GRID_CSV, mode="a", header=not GRID_CSV.exists(), index=False)
                    seen.add(key)
                    print(row["model"], task, k, r, d, ne, round(row["val_macro_f1"], 4), round(row["size_kb"]), round(row["lat_ms"], 3), flush=True)


def select():
    """moderate (<=1 pt val macro-F1 loss) and aggressive (<=3 pt): smallest size, tie -> lowest latency."""
    g = pd.read_csv(GRID_CSV)
    cfg = jload(RES / "baseline_config.json")
    sel = {}
    for task in ("binary", "multi"):
        Xva, yva = data.load("val", task)
        for name in ("dt", "rf", "xgb"):
            base = cfg[f"{name}_{task}"]["val_macro_f1"]
            d = g[(g.model == name) & (g.task == task)]
            for tag, tol in (("mod", 0.01), ("agg", 0.03)):
                ok = d[d.val_macro_f1 >= base - tol].sort_values(["size_kb", "lat_ms"])
                if ok.empty:
                    ok = d.sort_values("val_macro_f1", ascending=False).head(1)  # nothing within tol: best available
                r = ok.iloc[0]
                sel[f"{name}_{task}_{tag}"] = {k: (None if pd.isna(v) else v) for k, v in r.items()} | {"baseline_val_f1": base, "tol": tol}
    jdump(sel, RES / "tree_selected_variants.json")
    pd.DataFrame(sel).T.to_csv(RES / "tree_selected_variants.csv")
    return sel


def materialise_and_test(seed=SEED):
    """Train selected variants, save + register, score on test once each."""
    sel = jload(RES / "tree_selected_variants.json")
    for mid, r in sel.items():
        if (RES / f"metrics_{mid}.json").exists():
            continue
        name, task, _ = mid.split("_")
        k = r["k"] if r["k"] == "all" else int(r["k"])
        depth = None if r["depth"] in (None, "None") else int(float(r["depth"]))
        m, cols = fit(name, task, k, r["rank"], depth, int(r["n_est"]))
        if name == "xgb":
            p = MOD / f"{mid}.json"; m.get_booster().save_model(p)
        else:
            p = MOD / f"{mid}.joblib"; joblib.dump(m, p)
        predictors.register(mid, "xgb" if name == "xgb" else "sklearn", task, [p.name], cols=cols)
        Xte, yte = data.load("test", task)
        pred = predictors.load(mid)(Xte.to_numpy(np.float32))
        score_test(mid, task, yte, pred, {"n_features": len(cols)})


if __name__ == "__main__":
    {"grid": grid, "select": select, "final": materialise_and_test}[sys.argv[1]]()
