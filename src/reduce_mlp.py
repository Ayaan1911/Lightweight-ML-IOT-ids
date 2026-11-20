"""MLP reduction: pruning, int8 TFLite, distillation (+control), distilled+int8.  python -m src.reduce_mlp"""
import gzip, io, time, numpy as np, pandas as pd
from .common import *
set_single_thread_off = None  # training uses all cores; only bench.py forces 1 thread
import keras, tensorflow as tf
from . import data, predictors
from .training import scaled, fit_mlp, mlp_predict, build_mlp
from .evaluation import val_f1, score_test

SPARS = [0.5, 0.7, 0.9]
STUDENT, TEACHER = (32, 16), (256, 128, 64)


def mats(task):
    n_out = 2 if task == "binary" else len(CLASSES)
    Xtr, ytr = data.load("train", task); Xva, yva = data.load("val", task)
    return n_out, scaled(Xtr), ytr, scaled(Xva), yva


def save_keras(mid, task, m):
    p = MOD / f"{mid}.keras"; m.save(p)
    predictors.register(mid, "keras", task, [p.name], scaler=True)


def gz_kb(path):
    return len(gzip.compress(open(path, "rb").read(), 9)) / 1024


# ---------- (a) pruning ----------
def prune(base, sparsity, Xtr, ytr, Xva, yva, n_out, epochs=8):
    m = keras.models.clone_model(base); m.set_weights(base.get_weights())
    m.compile("adam", "binary_crossentropy" if n_out == 2 else "sparse_categorical_crossentropy")
    dense = [l for l in m.layers if isinstance(l, keras.layers.Dense)]
    masks = []
    for l in dense:
        w = l.get_weights()[0]
        thr = np.quantile(np.abs(w), sparsity)  # per-layer magnitude threshold
        masks.append((np.abs(w) > thr).astype(np.float32))

    def apply():
        for l, mk in zip(dense, masks):
            w, b = l.get_weights(); l.set_weights([w * mk, b])

    class Reapply(keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            apply()
    apply()
    m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=512, verbose=0, callbacks=[Reapply()])
    apply()
    return m, float(np.mean([1 - mk.mean() for mk in masks]))


# ---------- (b) quantisation ----------
def to_tflite(m, rep, dst, dynamic=False):
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    mode = "full-integer"
    try:
        if dynamic:
            raise RuntimeError("dynamic-range requested")
        conv.representative_dataset = lambda: ([rep[i:i + 1]] for i in range(len(rep)))
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        blob = conv.convert()
    except Exception as e:  # fall back to dynamic range
        print("full-integer failed:", repr(e)[:200], flush=True)
        conv = tf.lite.TFLiteConverter.from_keras_model(m); conv.optimizations = [tf.lite.Optimize.DEFAULT]
        blob = conv.convert(); mode = "dynamic-range"
    dst.write_bytes(blob)
    return mode


# ---------- (c) distillation ----------
def build_student(n_in, n_out, hidden, seed):
    keras.utils.set_random_seed(seed)
    inp = keras.Input((n_in,)); h = inp
    for u in hidden:
        h = keras.layers.Dense(u, activation="relu")(h)
    z = keras.layers.Dense(1 if n_out == 2 else n_out)(h)
    out = keras.layers.Activation("sigmoid" if n_out == 2 else "softmax")(z)
    return keras.Model(inp, out), keras.Model(inp, z)


def teacher_logits(t, X, n_out):
    p = np.clip(t.predict(X, batch_size=4096, verbose=0), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)) if n_out == 2 else np.log(p)


def distill(Xtr, ytr, Xva, yva, tl_tr, n_out, T, alpha, seed, hidden=STUDENT, epochs=30):
    full, logit = build_student(Xtr.shape[1], n_out, hidden, seed)

    def loss(yt, z):
        ops = keras.ops
        y = yt[:, 0]
        if n_out == 2:
            hard = keras.losses.binary_crossentropy(y[:, None], ops.sigmoid(z), from_logits=False)
            soft = keras.losses.binary_crossentropy(ops.sigmoid(yt[:, 1:2] / T), ops.sigmoid(z / T))
        else:
            hard = keras.losses.sparse_categorical_crossentropy(ops.cast(y, "int32"), z, from_logits=True)
            pt = ops.softmax(yt[:, 1:] / T); ps = ops.log_softmax(z / T)
            soft = ops.sum(pt * (ops.log(pt + 1e-9) - ps), axis=-1)
        return (1 - alpha) * hard + alpha * T * T * soft
    logit.compile("adam", loss)
    tgt = np.c_[ytr.astype(np.float32), tl_tr]
    full.compile("adam", "binary_crossentropy" if n_out == 2 else "sparse_categorical_crossentropy")
    yv_t = np.c_[yva.astype(np.float32), np.zeros((len(yva), 1 if n_out == 2 else n_out), np.float32)]
    logit.fit(Xtr, tgt, epochs=epochs, batch_size=512, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)])
    return full


def run(task):
    n_out, Xtr, ytr, Xva, yva = mats(task)
    base = keras.models.load_model(MOD / f"mlp_{task}.keras", compile=False)
    Xte_raw, yte = data.load("test", task)
    Xte = scaled(Xte_raw)
    rows = []

    def finish(mid, m_or_path, kind, extra):
        """freeze -> record val -> single test scoring."""
        if kind == "keras":
            pv = mlp_predict(m_or_path, Xva, n_out)
        else:
            pv = predictors.load(mid)(data.load("val", task)[0].to_numpy(np.float32))
        extra = dict(extra, val_macro_f1=val_f1(yva, pv))
        if kind == "keras":
            pt = predictors.load(mid)(Xte_raw.to_numpy(np.float32))
        else:
            pt = predictors.load(mid)(Xte_raw.to_numpy(np.float32))
        score_test(mid, task, yte, pt, extra)
        print(mid, extra, flush=True)

    # (a) pruning
    for s in SPARS:
        mid = f"mlp_{task}_prune{int(s*100)}"
        if (RES / f"metrics_{mid}.json").exists(): continue
        m, real = prune(base, s, Xtr, ytr, Xva, yva, n_out)
        save_keras(mid, task, m)
        raw = (MOD / f"{mid}.keras").stat().st_size / 1024
        finish(mid, m, "keras", {"sparsity": real, "size_raw_kb": raw, "size_gzip_kb": gz_kb(MOD / f"{mid}.keras")})
    # (b) int8 of the baseline
    rep = Xtr[np.random.default_rng(SEED).choice(len(Xtr), 500, replace=False)]
    def q8(mid, model, dynamic=False):
        p = MOD / f"{mid}.tflite"; mode = to_tflite(model, rep, p, dynamic)
        predictors.register(mid, "tflite", task, [p.name], scaler=True)
        return mode
    mid = f"mlp_{task}_q8"
    if not (RES / f"metrics_{mid}.json").exists():
        mode = q8(mid, base)
        finish(mid, None, "tflite", {"quant_mode": mode})
    # (c) teacher, student grid on validation, control
    tpath = MOD / f"teacher_{task}.keras"
    if tpath.exists():
        teacher = keras.models.load_model(tpath, compile=False)
    else:
        teacher = fit_mlp(Xtr, ytr, Xva, yva, n_out, hidden=TEACHER); teacher.save(tpath)
    tl = teacher_logits(teacher, Xtr, n_out)
    cfgp = RES / f"distill_search_{task}.json"
    if not cfgp.exists():
        res = []
        for T, a in ((2, 0.5), (4, 0.5), (4, 0.9)):
            st = distill(Xtr, ytr, Xva, yva, tl, n_out, T, a, SEED)
            res.append({"T": T, "alpha": a, "val_macro_f1": val_f1(yva, mlp_predict(st, Xva, n_out))})
        ctrl = distill(Xtr, ytr, Xva, yva, tl, n_out, 1, 0.0, SEED)  # alpha=0 -> hard labels only (same code path)
        best = max(res, key=lambda r: r["val_macro_f1"])
        jdump({"grid": res, "best": best, "control_val_macro_f1": val_f1(yva, mlp_predict(ctrl, Xva, n_out)),
               "teacher_val_macro_f1": val_f1(yva, mlp_predict(teacher, Xva, n_out))}, cfgp)
    best = jload(cfgp)["best"]
    mid = f"mlp_{task}_ctrl"
    if not (RES / f"metrics_{mid}.json").exists():
        st = distill(Xtr, ytr, Xva, yva, tl, n_out, 1, 0.0, SEED); save_keras(mid, task, st)
        finish(mid, st, "keras", {"student": str(STUDENT)})
    mid = f"mlp_{task}_dist"
    if not (RES / f"metrics_{mid}.json").exists():
        st = distill(Xtr, ytr, Xva, yva, tl, n_out, best["T"], best["alpha"], SEED); save_keras(mid, task, st)
        finish(mid, st, "keras", {"student": str(STUDENT), "T": best["T"], "alpha": best["alpha"]})
    # (d) distilled + int8
    mid = f"mlp_{task}_distq8"
    if not (RES / f"metrics_{mid}.json").exists():
        st = keras.models.load_model(MOD / f"mlp_{task}_dist.keras", compile=False)
        mode = q8(mid, st); finish(mid, None, "tflite", {"quant_mode": mode})
    # dynamic-range alternatives (chosen vs full-integer on validation in src/quant_choice.py)
    mid = f"mlp_{task}_q8dyn"
    if not (RES / f"metrics_{mid}.json").exists():
        finish(mid, None, "tflite", {"quant_mode": q8(mid, base, True)})
    mid = f"mlp_{task}_distq8dyn"
    if not (RES / f"metrics_{mid}.json").exists():
        st = keras.models.load_model(MOD / f"mlp_{task}_dist.keras", compile=False)
        finish(mid, None, "tflite", {"quant_mode": q8(mid, st, True)})


if __name__ == "__main__":
    for t in ("binary", "multi"):
        run(t)
