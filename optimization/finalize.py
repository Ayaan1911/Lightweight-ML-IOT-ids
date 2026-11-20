"""Retrain every optimizer winner on the full training split, register it, score it ONCE on test.
python -m optimization.finalize     (resumable: skips model ids that already have metrics)"""
import os, sys, time, json, glob, joblib, numpy as np, pandas as pd
os.environ.update(TF_CPP_MIN_LOG_LEVEL="3", TF_ENABLE_ONEDNN_OPTS="0")
from src.common import ROOT, MOD, CLASSES, jdump, jload, log_test_access
from src import data, predictors
from src.evaluation import compute
from src.training import get_scaler
from optimization import core

OUT = ROOT / "results" / "optimization"; FIN = OUT / "final"
SEEDS = (42, 43, 44)
METHOD_ORDER = ("ga", "pso", "aco", "random")


def score(mid, task, y, pred, extra):
    n = 2 if task == "binary" else len(CLASSES)
    log_test_access(mid)  # every test use is logged
    np.savez(FIN / f"pred_{mid}.npz", y_true=y, y_pred=pred)
    m = compute(y, pred, n)
    names = ["Benign", "Attack"] if task == "binary" else CLASSES
    pd.DataFrame(m["cm"], index=names, columns=names).to_csv(FIN / f"cm_{mid}.csv")
    row = {"model_id": mid, "task": task, **{k: v for k, v in m.items() if k not in ("cm", "per_class_f1")}, **extra}
    if task == "multi":
        row.update({f"f1_{c}": f for c, f in zip(CLASSES, m["per_class_f1"])})
    jdump(row, FIN / f"metrics_{mid}.json")


def build(run, seed, mid):
    model, task = run["model"], run["task"]
    n_out = 2 if task == "binary" else len(CLASSES)
    cols = [i for i, b in enumerate(run["mask"]) if b]
    Xtr, ytr = data.load("train", task); Xva, yva = data.load("val", task)
    A, B = Xtr.to_numpy(np.float32)[:, cols], Xva.to_numpy(np.float32)[:, cols]
    if model == "mlp":
        sc = get_scaler(); mu, sd = sc.mean_.astype(np.float32)[cols], sc.scale_.astype(np.float32)[cols]
        A, B = (A - mu) / sd, (B - mu) / sd
    t = time.time()
    pred, km = core.fit_model(model, run["hp"], A, ytr, n_out, seed, 4, B, yva)
    train_s = time.time() - t
    val_f1 = float(compute(yva, pred(B), n_out)["macro_f1"])  # full validation split (allowed)
    if model in ("dt", "rf"):
        p = MOD / f"{mid}.joblib"; joblib.dump(pred.__self__, p); kind = "sklearn"
    elif model == "xgb":
        p = MOD / f"{mid}.json"; km.get_booster().save_model(p); kind = "xgb"
    else:
        p = MOD / f"{mid}.keras"; km.save(p); kind = "keras"
        import keras; keras.backend.clear_session()
    predictors.register(mid, kind, task, [p.name], cols=cols, scaler=(model == "mlp"))
    return cols, train_s, val_f1


def main():
    FIN.mkdir(exist_ok=True)
    runs = [jload(f) for f in sorted(glob.glob(str(OUT / "runs" / "*.json")))]
    for seed in SEEDS:
        for run in runs:
            base = f"{run['model']}_{run['task']}_{run['method']}"
            mid = base if seed == 42 else f"{base}_s{seed}"
            if (FIN / f"metrics_{mid}.json").exists():
                continue
            cols, train_s, val_f1 = build(run, seed, mid)
            Xte, yte = data.load("test", run["task"])
            pred = predictors.load(mid)(Xte.to_numpy(np.float32))  # same path as the benchmark
            score(mid, run["task"], yte, pred, {"optimizer": run["method"], "model": run["model"], "seed": seed, "n_features": len(cols),
                                               "train_s": train_s, "val_macro_f1_full": val_f1})
            jdump({"model_id": mid, "cols": cols, "hp": run["hp"], "seed": seed}, FIN / f"config_{mid}.json")
            print(mid, round(val_f1, 4), flush=True)


if __name__ == "__main__":
    main()
