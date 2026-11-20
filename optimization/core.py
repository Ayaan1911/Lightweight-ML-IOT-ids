"""Candidate encoding + fitness for the metaheuristic phase.
This module reads TRAIN and VALIDATION rows only (the loader below never touches the held-out split)."""
import gc, time, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from src.common import PROC, MOD, CLASSES, SEED

N_FEAT, MIN_FEAT, LAMBDA = 39, 5, 0.001
N_TR_SUB, N_VA_SUB = 100_000, 50_000
HP = {  # (name, kind, lo, hi) or (name, "cat", choices)
    "dt": [("max_depth", "int", 3, 40), ("min_samples_split", "int", 2, 50), ("min_samples_leaf", "int", 1, 30),
           ("criterion", "cat", ["gini", "entropy", "log_loss"]), ("class_weight", "cat", [None, "balanced"])],
    "rf": [("n_estimators", "int", 10, 150), ("max_depth", "int", 3, 30), ("min_samples_leaf", "int", 1, 20),
           ("max_features", "cat", ["sqrt", "log2", 0.5]), ("class_weight", "cat", [None, "balanced", "balanced_subsample"])],
    "xgb": [("n_estimators", "int", 20, 200), ("max_depth", "int", 2, 10), ("learning_rate", "log", 0.02, 0.4),
            ("subsample", "float", 0.5, 1.0), ("colsample_bytree", "float", 0.5, 1.0), ("min_child_weight", "int", 1, 10)],
    "mlp": [("n_layers", "int", 1, 3), ("u1", "int", 16, 128), ("u2", "int", 16, 128), ("u3", "int", 16, 128),
            ("lr", "log", 1e-4, 1e-2), ("dropout", "float", 0.0, 0.4), ("batch", "cat", [256, 512, 1024, 2048])],
}


def dim(model):
    return N_FEAT + len(HP[model])


def decode(model, x):
    """x in [0,1]^dim -> (bool mask [N_FEAT], effective hyperparameter dict). Deterministic; defines candidate identity."""
    x = np.clip(np.asarray(x, float), 0, 1)
    mask = x[:N_FEAT] > 0.5
    if mask.sum() < MIN_FEAT:
        mask = np.zeros(N_FEAT, bool); mask[np.argsort(-x[:N_FEAT])[:MIN_FEAT]] = True
    hp = {}
    for v, spec in zip(x[N_FEAT:], HP[model]):
        n, kind = spec[0], spec[1]
        if kind == "int": hp[n] = int(round(spec[2] + v * (spec[3] - spec[2])))
        elif kind == "float": hp[n] = float(round(spec[2] + v * (spec[3] - spec[2]), 4))
        elif kind == "log": hp[n] = float(f"{np.exp(np.log(spec[2]) + v * (np.log(spec[3]) - np.log(spec[2]))):.4g}")
        else: hp[n] = spec[2][min(int(v * len(spec[2])), len(spec[2]) - 1)]
    if model == "mlp":
        hid = [hp.pop(f"u{i}") for i in (1, 2, 3)]
        hp["hidden"] = hid[:hp.pop("n_layers")]
    return mask, hp


def key(model, x):
    mask, hp = decode(model, x)
    return (mask.tobytes(), tuple(sorted((k, str(v)) for k, v in hp.items())))


_D = {}


def data(task):
    """Fixed stratified subsamples (train 100k / val 50k, rare classes kept) + training-fitted scaler. Train/val only."""
    if task in _D:
        return _D[task]
    df = pd.read_parquet(PROC / "clean.parquet"); ix = np.load(PROC / "split.npz")
    feat = [c for c in df.columns if not c.startswith("y_")]
    out = {}
    for name, n in (("train", N_TR_SUB), ("val", N_VA_SUB)):
        d = df.iloc[ix[name]]
        y = (d["y_binary"].to_numpy() if task == "binary" else d["y_multi"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()).astype("int64")
        rng = np.random.default_rng(SEED); take = []
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)
            k = min(len(idx), max(int(round(n * len(idx) / len(y))), 200 if name == "train" else 100))
            take.append(rng.choice(idx, k, replace=False))
        take = rng.permutation(np.concatenate(take))
        out[name] = (d[feat].to_numpy(np.float32)[take], y[take])
    del df; gc.collect()
    z = np.load(MOD / "scaler.npz")
    out["mu"], out["sd"] = z["mean"].astype(np.float32), z["scale"].astype(np.float32)
    _D[task] = out
    return out


def make_mlp(n_in, n_out, hp, seed):
    import keras
    keras.utils.set_random_seed(seed)
    L = [keras.layers.Input((n_in,))]
    for h in hp["hidden"]:
        L.append(keras.layers.Dense(h, activation="relu"))
        if hp["dropout"] > 0: L.append(keras.layers.Dropout(hp["dropout"]))
    L.append(keras.layers.Dense(1, activation="sigmoid") if n_out == 2 else keras.layers.Dense(n_out, activation="softmax"))
    m = keras.Sequential(L)
    m.compile(keras.optimizers.Adam(hp["lr"]), "binary_crossentropy" if n_out == 2 else "sparse_categorical_crossentropy")
    return m


def fit_model(model, hp, Xtr, ytr, n_out, seed, nj, Xva=None, yva=None):
    """Train one model on (already column-selected) Xtr; returns (predict callable, keras model or None).
    MLP: Xtr must be standardised; early stopping on (Xva,yva) if given, else on a 10% hold-out of Xtr."""
    if model == "dt":
        return DecisionTreeClassifier(random_state=seed, **hp).fit(Xtr, ytr).predict, None
    if model == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=nj, **hp).fit(Xtr, ytr).predict, None
    if model == "xgb":
        m = xgb.XGBClassifier(tree_method="hist", n_jobs=nj, random_state=seed, **hp).fit(Xtr, ytr)
        return m.predict, m
    import keras
    m = make_mlp(Xtr.shape[1], n_out, hp, seed)
    kw = dict(validation_data=(Xva, yva)) if Xva is not None else dict(validation_split=0.1)
    m.fit(Xtr, ytr, epochs=10, batch_size=hp["batch"], verbose=0, **kw,
          callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])

    def pred(X):
        p = m.predict(X, batch_size=4096, verbose=0)
        return (p[:, 0] > 0.5).astype(int) if n_out == 2 else p.argmax(1)
    return pred, m


def fitness(model, task, x, nj=4):
    """macro-F1(val subsample) - LAMBDA * fraction of features, for a model trained on the train subsample."""
    d = data(task); mask, hp = decode(model, x); n_out = 2 if task == "binary" else len(CLASSES)
    Xtr, ytr = d["train"]; Xva, yva = d["val"]; cols = np.flatnonzero(mask)
    A, B = Xtr[:, cols], Xva[:, cols]
    if model == "mlp":
        A, B = (A - d["mu"][cols]) / d["sd"][cols], (B - d["mu"][cols]) / d["sd"][cols]
    t = time.time()
    pred, _ = fit_model(model, hp, A, ytr, n_out, SEED, nj)
    f1 = float(f1_score(yva, pred(B), average="macro", zero_division=0))
    if model == "mlp":
        import keras; keras.backend.clear_session()
    gc.collect()
    return f1 - LAMBDA * mask.mean(), {"val_sub_macro_f1": f1, "n_features": int(mask.sum()), "eval_s": time.time() - t}
