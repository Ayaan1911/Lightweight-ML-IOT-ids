"""Baseline training: light val-only tuning with a time budget. Run: python -m src.training"""
import time, joblib, numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import xgboost as xgb
from .common import *
from . import data, predictors
from .evaluation import val_f1, score_test

BUDGET_S = 420  # per model per task


def get_scaler():
    p = MOD / "scaler.joblib"
    if p.exists():
        return joblib.load(p)
    Xtr, _ = data.load("train")
    sc = StandardScaler().fit(Xtr.to_numpy(np.float64))
    joblib.dump(sc, p)
    np.savez(MOD / "scaler.npz", mean=sc.mean_, scale=sc.scale_)
    return sc


def scaled(X):
    sc = get_scaler()
    return ((X.to_numpy(np.float32) - sc.mean_.astype(np.float32)) / sc.scale_.astype(np.float32)).astype(np.float32)


def build_mlp(n_in, n_out, hidden=(128, 64), seed=SEED):
    import keras
    keras.utils.set_random_seed(seed)
    layers = [keras.layers.Input((n_in,))] + [keras.layers.Dense(h, activation="relu") for h in hidden]
    layers.append(keras.layers.Dense(1, activation="sigmoid") if n_out == 2 else keras.layers.Dense(n_out, activation="softmax"))
    m = keras.Sequential(layers)
    m.compile("adam", "binary_crossentropy" if n_out == 2 else "sparse_categorical_crossentropy")
    return m


def fit_mlp(Xtr, ytr, Xva, yva, n_out, hidden=(128, 64), weighted=False, epochs=30, seed=SEED):
    import keras
    m = build_mlp(Xtr.shape[1], n_out, hidden, seed)
    cw = None
    if weighted:
        cl = np.unique(ytr)
        cw = dict(zip(cl, compute_class_weight("balanced", classes=cl, y=ytr)))
    m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=512, class_weight=cw, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    return m


def mlp_predict(m, X, n_out):
    pr = m.predict(X, batch_size=4096, verbose=0)
    return (pr[:, 0] > 0.5).astype(int) if n_out == 2 else pr.argmax(1)


def candidates(name, n_out):
    if name == "dt":
        return [dict(class_weight=cw, max_depth=md) for cw in (None, "balanced") for md in (None, 30, 20)]
    if name == "rf":
        return [dict(class_weight=cw, n_estimators=100) for cw in (None, "balanced_subsample")]
    if name == "xgb":
        return [dict(weighted=w, max_depth=md, n_estimators=ne, learning_rate=lr)
                for w in (False, True) for md, ne, lr in ((6, 150, 0.2), (10, 150, 0.2), (10, 300, 0.1), (14, 200, 0.1))]
    if name == "mlp":
        return [dict(weighted=w, hidden=(128, 64)) for w in (False, True)]


def train_one(name, cand, Xtr, ytr, Xva, n_out, seed=SEED):
    """Return (model, predict_fn(X_df))."""
    if name == "dt":
        m = DecisionTreeClassifier(random_state=seed, **cand).fit(Xtr, ytr)
        return m, m.predict
    if name == "rf":
        m = RandomForestClassifier(random_state=seed, n_jobs=-1, **cand).fit(Xtr, ytr)
        return m, m.predict
    if name == "xgb":
        c = dict(cand); w = c.pop("weighted")
        m = xgb.XGBClassifier(tree_method="hist", n_jobs=-1, random_state=seed, **c)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr) if w else None)
        return m, m.predict
    raise ValueError(name)


def save_model(name, task, m):
    if name in ("dt", "rf"):
        p = MOD / f"{name}_{task}.joblib"
        joblib.dump(m, p)
    elif name == "xgb":
        p = MOD / f"{name}_{task}.json"
        m.get_booster().save_model(p)
    else:
        p = MOD / f"{name}_{task}.keras"
        m.save(p)
    kind = {"dt": "sklearn", "rf": "sklearn", "xgb": "xgb", "mlp": "keras"}[name]
    predictors.register(f"{name}_{task}", kind, task, [p.name], scaler=(name == "mlp"))
    return p


def run_baselines():
    cfg_path = RES / "baseline_config.json"
    cfg = jload(cfg_path) if cfg_path.exists() else {}
    for task in ("binary", "multi"):
        n_out = 2 if task == "binary" else len(CLASSES)
        Xtr, ytr = data.load("train", task)
        Xva, yva = data.load("val", task)
        for name in ("dt", "rf", "xgb", "mlp"):
            mid = f"{name}_{task}"
            if mid in cfg:
                continue
            t_start = time.time(); trials = []; best = None
            for cand in candidates(name, n_out):
                if time.time() - t_start > BUDGET_S and best is not None:
                    break
                t0 = time.time()
                if name == "mlp":
                    m = fit_mlp(scaled(Xtr), ytr, scaled(Xva), yva, n_out, weighted=cand["weighted"], hidden=cand["hidden"])
                    pv = mlp_predict(m, scaled(Xva), n_out)
                else:
                    m, pf = train_one(name, cand, Xtr, ytr, Xva, n_out)
                    pv = pf(Xva)
                f = val_f1(yva, pv)
                trials.append({"cand": cand, "val_macro_f1": f, "train_s": time.time() - t0})
                print(mid, cand, round(f, 4), round(time.time() - t0, 1), flush=True)
                if best is None or f > best[0]:
                    best = (f, cand, m, time.time() - t0)
            f, cand, m, ts = best
            p = save_model(name, task, m)
            cfg[mid] = {"chosen": cand, "val_macro_f1": f, "train_time_s": ts, "trials": trials, "file": p.name, "seed": SEED,
                        "n_features": Xtr.shape[1]}
            jdump(cfg, cfg_path)  # frozen
    # single test scoring per frozen model
    for mid in cfg:
        if (RES / f"metrics_{mid}.json").exists():
            continue
        name, task = mid.split("_")
        yte = data.load("test", task)[1]
        Xte = data.load("test", task)[0]
        n_out = 2 if task == "binary" else len(CLASSES)
        pred = predictors.load(mid)(Xte.to_numpy(np.float32))  # same code path as the benchmark
        score_test(mid, task, yte, pred)


def predict_saved(name, task, X):
    n_out = 2 if task == "binary" else len(CLASSES)
    if name in ("dt", "rf"):
        return joblib.load(MOD / f"{name}_{task}.joblib").predict(X)
    if name == "xgb":
        b = xgb.Booster(); b.load_model(MOD / f"xgb_{task}.json")
        pr = b.predict(xgb.DMatrix(X.to_numpy()))
        return (pr > 0.5).astype(int) if task == "binary" else pr.argmax(1) if pr.ndim == 2 else pr.astype(int)
    import keras
    return mlp_predict(keras.models.load_model(MOD / f"mlp_{task}.keras"), scaled(X), n_out)


if __name__ == "__main__":
    run_baselines()
