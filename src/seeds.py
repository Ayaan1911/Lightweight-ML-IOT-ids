"""Retrain final configurations with seeds 43, 44 (split fixed); each seed model scored on test once.
python -m src.seeds   ->  results/seed_stability.csv"""
import joblib, numpy as np, pandas as pd
import keras, tensorflow as tf
from .common import *
from . import data, predictors, reduce_trees as rt, reduce_mlp as rm
from .training import train_one, scaled, fit_mlp, save_model
from .evaluation import score_test

SEEDS = (43, 44)


def done(mid):
    return (RES / f"metrics_{mid}.json").exists()


def score(mid, task, extra=None):
    Xte, yte = data.load("test", task)
    score_test(mid, task, yte, predictors.load(mid)(Xte.to_numpy(np.float32)), extra)


def run():
    cfg = jload(RES / "baseline_config.json"); sel = jload(RES / "tree_selected_variants.json")
    qc = jload(RES / "mlp_quant_choice.json")
    for task in ("binary", "multi"):
        n_out = 2 if task == "binary" else len(CLASSES)
        Xtr, ytr = data.load("train", task); Xva, yva = data.load("val", task)
        S_tr, S_va = scaled(Xtr), scaled(Xva)
        rep = S_tr[np.random.default_rng(SEED).choice(len(S_tr), 500, replace=False)]
        teacher = keras.models.load_model(MOD / f"teacher_{task}.keras", compile=False)
        tl = rm.teacher_logits(teacher, S_tr, n_out)
        best = jload(RES / f"distill_search_{task}.json")["best"]
        for s in SEEDS:
            for name in ("dt", "rf", "xgb"):  # baselines
                mid = f"{name}_{task}_s{s}"
                if not done(mid):
                    m, _ = train_one(name, cfg[f"{name}_{task}"]["chosen"], Xtr, ytr, Xva, n_out, seed=s)
                    p = save_model(name, task, m) if False else None
                    if name == "xgb":
                        p = MOD / f"{mid}.json"; m.get_booster().save_model(p)
                    else:
                        p = MOD / f"{mid}.joblib"; joblib.dump(m, p)
                    predictors.register(mid, "xgb" if name == "xgb" else "sklearn", task, [p.name])
                    score(mid, task, {"seed": s})
                r = sel[f"{name}_{task}_mod"]  # moderate variant
                mid = f"{name}_{task}_mod_s{s}"
                if not done(mid):
                    k = r["k"] if r["k"] == "all" else int(r["k"])
                    d = None if r["depth"] is None else int(float(r["depth"]))
                    m, cols = rt.fit(name, task, k, r["rank"], d, int(r["n_est"]), seed=s)
                    if name == "xgb":
                        p = MOD / f"{mid}.json"; m.get_booster().save_model(p)
                    else:
                        p = MOD / f"{mid}.joblib"; joblib.dump(m, p)
                    predictors.register(mid, "xgb" if name == "xgb" else "sklearn", task, [p.name], cols=cols)
                    score(mid, task, {"seed": s})
            # MLP: baseline, chosen-quantised baseline, distilled student, control student
            c = cfg[f"mlp_{task}"]["chosen"]
            mid = f"mlp_{task}_s{s}"
            base = None
            if not done(mid):
                base = fit_mlp(S_tr, ytr, S_va, yva, n_out, hidden=tuple(c["hidden"]), weighted=c["weighted"], seed=s)
                rm.save_keras(mid, task, base); score(mid, task, {"seed": s})
            mid = f"mlp_{task}_q8_s{s}"
            if not done(mid):
                base = base or keras.models.load_model(MOD / f"mlp_{task}_s{s}.keras", compile=False)
                dyn = qc[f"mlp_{task}_q8"]["chosen"].endswith("dyn")
                conv = tf.lite.TFLiteConverter.from_keras_model(base); conv.optimizations = [tf.lite.Optimize.DEFAULT]
                if not dyn:
                    conv.representative_dataset = lambda: ([rep[i:i + 1]] for i in range(len(rep)))
                    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                (MOD / f"{mid}.tflite").write_bytes(conv.convert())
                predictors.register(mid, "tflite", task, [f"{mid}.tflite"], scaler=True); score(mid, task, {"seed": s})
            for tag, (T, a) in (("dist", (best["T"], best["alpha"])), ("ctrl", (1, 0.0))):
                mid = f"mlp_{task}_{tag}_s{s}"
                if not done(mid):
                    st = rm.distill(S_tr, ytr, S_va, yva, tl, n_out, T, a, s)
                    rm.save_keras(mid, task, st); score(mid, task, {"seed": s})


def summarise():
    rows = []
    for task in ("binary", "multi"):
        cfgs = {"dt": f"dt_{task}", "rf": f"rf_{task}", "xgb": f"xgb_{task}", "dt_mod": f"dt_{task}_mod", "rf_mod": f"rf_{task}_mod",
                "xgb_mod": f"xgb_{task}_mod", "mlp": f"mlp_{task}", "mlp_q8(chosen)": f"mlp_{task}_q8", "mlp_dist": f"mlp_{task}_dist",
                "mlp_ctrl": f"mlp_{task}_ctrl"}
        for label, base in cfgs.items():
            ids = [base] + [f"{base}_s{s}" for s in SEEDS]
            if label.startswith("mlp_q8"):
                ids = [jload(RES / "mlp_quant_choice.json")[f"mlp_{task}_q8"]["chosen"]] + [f"mlp_{task}_q8_s{s}" for s in SEEDS]
            v = [jload(RES / f"metrics_{i}.json")["macro_f1"] for i in ids if done(i)]
            rows.append({"task": task, "config": label, "n_seeds": len(v), "macro_f1_mean": np.mean(v), "macro_f1_std": np.std(v, ddof=1) if len(v) > 1 else np.nan,
                         "values": str([round(x, 4) for x in v])})
    pd.DataFrame(rows).to_csv(RES / "seed_stability.csv", index=False)


if __name__ == "__main__":
    run(); summarise()
