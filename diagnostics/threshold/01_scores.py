"""Predict-only: attack scores (P(Attack)) of the 4 saved binary baselines on val and test. No refit, nothing written outside this folder.
Run from project root: .venv\Scripts\python diagnostics\threshold\01_scores.py"""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.common import set_single_thread, MOD
set_single_thread()
import numpy as np, joblib, xgboost as xgb
from src import data, predictors

OUT = Path(__file__).parent
Xs = {s: data.load(s, "binary") for s in ("val", "test")}
mu, sd = predictors._scaler()


def score(name, X):
    f = MOD / predictors.meta(f"{name}_binary")["files"][0]
    if name in ("dt", "rf"):
        m = joblib.load(f)
        if hasattr(m, "n_jobs"): m.n_jobs = 1
        return m.predict_proba(X)[:, 1]
    if name == "xgb":
        b = xgb.Booster(); b.load_model(f); b.set_param({"nthread": 1}); b.feature_names = None; b.feature_types = None
        return b.inplace_predict(X).astype(np.float64)
    import keras
    m = keras.models.load_model(f, compile=False)
    return m(((X - mu) / sd).astype(np.float32), training=False).numpy()[:, 0].astype(np.float64)


for n in ("dt", "rf", "xgb", "mlp"):
    arr = {}
    for s, (X, y) in Xs.items():
        arr[f"{s}_score"] = score(n, np.ascontiguousarray(X.to_numpy(np.float32))); arr[f"{s}_y"] = y
    np.savez_compressed(OUT / f"scores_{n}.npz", **arr)
    print(n, {k: v.shape for k, v in arr.items()})
