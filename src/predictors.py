"""Uniform loading/prediction for saved models via models/<id>.meta.json.
meta = {kind: sklearn|xgb|keras|tflite, task, files: [...], cols: [feature idx] or null, scaler: bool}"""
import numpy as np, joblib
from .common import *


def register(mid, kind, task, files, cols=None, scaler=False):
    jdump({"kind": kind, "task": task, "files": files, "cols": cols, "scaler": scaler}, MOD / f"{mid}.meta.json")


def meta(mid):
    return jload(MOD / f"{mid}.meta.json")


def size_kb(mid):
    m = meta(mid)
    files = list(m["files"]) + (["scaler.npz"] if m["scaler"] else [])
    return sum((MOD / f).stat().st_size for f in files) / 1024


def _scaler():
    z = np.load(MOD / "scaler.npz")
    return z["mean"].astype(np.float32), z["scale"].astype(np.float32)


def load(mid, threads=1):
    """Return predict(X float32 ndarray [n, n_all_features]) -> int labels."""
    m = meta(mid)
    cols = m["cols"]
    f0 = MOD / m["files"][0]
    if m["kind"] == "sklearn":
        mdl = joblib.load(f0)
        if hasattr(mdl, "n_jobs"):
            mdl.n_jobs = threads
        core = mdl.predict
    elif m["kind"] == "xgb":
        import xgboost as xgb
        b = xgb.Booster(); b.load_model(f0); b.set_param({"nthread": threads}); b.feature_names = None; b.feature_types = None
        binary = m["task"] == "binary"

        def core(x):
            pr = b.inplace_predict(x)
            return (pr > 0.5).astype(np.int64) if binary else pr.argmax(1)
    elif m["kind"] == "keras":
        import keras
        mdl = keras.models.load_model(f0, compile=False)
        binary = m["task"] == "binary"

        def core(x):
            pr = mdl(x, training=False).numpy()
            return (pr[:, 0] > 0.5).astype(np.int64) if binary else pr.argmax(1)
    elif m["kind"] == "tflite":
        import tensorflow as tf
        it = tf.lite.Interpreter(model_path=str(f0), num_threads=threads)
        it.allocate_tensors()
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        binary = m["task"] == "binary"
        cur = [int(i["shape"][0])]

        def core(x):
            if x.shape[0] != cur[0]:
                it.resize_tensor_input(i["index"], [x.shape[0], x.shape[1]]); it.allocate_tensors(); cur[0] = x.shape[0]
            it.set_tensor(i["index"], x.astype(np.float32)); it.invoke()
            pr = it.get_tensor(o["index"])
            return (pr[:, 0] > 0.5).astype(np.int64) if binary else pr.argmax(1)
    else:
        raise ValueError(m["kind"])
    if m["scaler"]:
        mu, sd = _scaler()

    def predict(X):
        if cols is not None:
            X = X[:, cols]
        if m["scaler"]:
            X = (X - (mu[cols] if cols else mu)) / (sd[cols] if cols else sd)
        return core(np.ascontiguousarray(X, dtype=np.float32))
    return predict
