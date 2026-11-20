"""Verification loop: python verify.py   (checks whatever artifacts exist; exit 1 on failure)."""
import sys, os, glob, json, numpy as np, pandas as pd
from pathlib import Path
from src.common import *
from src import data, predictors
from src.evaluation import compute

FAILS, PASSES = [], 0


def check(name, cond, detail=""):
    global PASSES
    if cond:
        PASSES += 1
    else:
        FAILS.append(f"{name} {detail}")
        print("FAIL:", name, detail)


def own_macro_f1(y, p, n):
    f = []
    for c in range(n):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def split_checks():
    if not (PROC / "split.npz").exists():
        return
    ix = np.load(PROC / "split.npz"); df = pd.read_parquet(PROC / "clean.parquet")
    tr, va, te = ix["train"], ix["val"], ix["test"]; n = len(df)
    check("split disjoint", not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te)))
    check("split covers all rows", len(tr) + len(va) + len(te) == n)
    check("split 70/15/15", abs(len(tr) / n - .70) < .001 and abs(len(va) / n - .15) < .001 and abs(len(te) / n - .15) < .001)
    for col in ("y_multi", "y_binary"):
        p = [df[col].iloc[s].value_counts(normalize=True) for s in (tr, va, te)]
        d = max(abs(p[0] - p[i]).max() for i in (1, 2)) * 100
        check(f"class proportions within 0.2pt ({col})", d < 0.2, f"max diff {d:.3f}pt")
    feat = [c for c in df.columns if not c.startswith("y_")]
    h = pd.util.hash_pandas_object(df[feat], index=False).to_numpy()
    s = [set(h[i]) for i in (tr, va, te)]
    check("no identical feature rows across splits", not (s[0] & s[1] or s[0] & s[2] or s[1] & s[2]))
    check("no duplicate feature rows at all", len(set(h)) == len(h))
    if (MOD / "scaler.npz").exists():
        z = np.load(MOD / "scaler.npz"); X = df.iloc[tr][feat].to_numpy(np.float64)
        check("scaler mean == train mean", np.allclose(z["mean"], X.mean(0), rtol=1e-9, atol=1e-9))
        check("scaler scale == train std", np.allclose(z["scale"], X.std(0), rtol=1e-9, atol=1e-9))


def test_log_checks():
    lg = ROOT / "test_access.log"
    ids = [l.split("\t")[0:2] for l in lg.read_text().splitlines()] if lg.exists() else []
    seen_all = pd.Series([i[1] for i in ids]).value_counts()
    check("each model scored on test exactly once", (seen_all == 1).all() if len(seen_all) else True, str(seen_all[seen_all > 1].to_dict()))
    opt_ids = {Path(f).stem[len("metrics_"):] for f in glob.glob(str(RES / "optimization" / "final" / "metrics_*.json"))}
    seen = seen_all[[i for i in seen_all.index if i not in opt_ids]]  # optimization-phase ids have their own check
    mids = {Path(f).stem[len("metrics_"):] for f in glob.glob(str(RES / "metrics_*.json"))}
    check("metrics files <-> test log", mids == set(seen.index), str(mids ^ set(seen.index)))
    # frozen before test: model file must not be newer than its test-log entry
    for t, mid in ids:
        mf = [p for p in MOD.glob(f"{mid}.*") if p.suffix != ".json" or p.name.endswith(".meta.json") is False]
        ts = pd.Timestamp(t).timestamp()
        late = [p.name for p in mf if p.stat().st_mtime > ts + 2]
        check(f"{mid} frozen before test scoring", not late, str(late))


def metric_checks():
    for f in glob.glob(str(RES / "metrics_*.json")):
        m = jload(f); mid = m["model_id"]; pz = RES / f"pred_{mid}.npz"
        if not pz.exists():
            check(f"{mid} predictions saved", False); continue
        z = np.load(pz); y, p = z["y_true"], z["y_pred"]; n = 2 if m["task"] == "binary" else len(CLASSES)
        r = compute(y, p, n)
        for k in ("macro_f1", "accuracy", "precision", "recall", "fpr"):
            check(f"{mid} {k} recompute", abs(r[k] - m[k]) < 1e-9)
        check(f"{mid} macro_f1 independent", abs(own_macro_f1(y, p, n) - m["macro_f1"]) < 1e-9)
        cm = pd.read_csv(RES / f"cm_{mid}.csv", index_col=0).to_numpy()
        check(f"{mid} cm sums to test size", cm.sum() == len(y) == len(data.load("test", m["task"])[1]))
        if n == 2:
            tn, fp = cm[0, 0], cm[0, 1]
            check(f"{mid} FPR formula", abs(m["fpr"] - fp / (fp + tn)) < 1e-12)
        else:
            fpr = np.mean([(cm[:, c].sum() - cm[c, c]) / (cm.sum() - cm[c, :].sum()) for c in range(n)])
            check(f"{mid} FPR formula (macro OvR)", abs(m["fpr"] - fpr) < 1e-12)
        if m["macro_f1"] > 0.99:
            print("NOTE high macro F1 >0.99:", mid, "-> run leakage checks (leakage_checks.py)")
    # test-set labels identical across models of a task
    for task in ("binary", "multi"):
        ys = [np.load(f)["y_true"] for f in glob.glob(str(RES / "pred_*.npz")) if jload(RES / f"metrics_{Path(f).stem[5:]}.json")["task"] == task]
        check(f"same test labels across {task} models", all(np.array_equal(ys[0], y) for y in ys) if ys else True)


def model_size_checks():
    for f in glob.glob(str(MOD / "*.meta.json")):
        mid = Path(f).name[:-10]; m = jload(f)
        exp = sum((MOD / x).stat().st_size for x in m["files"] + (["scaler.npz"] if m["scaler"] else [])) / 1024
        check(f"{mid} size_kb == getsize", abs(predictors.size_kb(mid) - exp) < 1e-9)
    for f in glob.glob(str(RES / "benchmark_*.csv")):
        b = pd.read_csv(f)
        b = b[b.get("failed", pd.Series(False, index=b.index)).fillna(False) == False] if "failed" in b else b
        for _, r in b.iterrows():
            if r.get("oom") is True:
                continue
            check(f"{Path(f).stem}:{r.model_id} size matches file", abs(r.size_kb - predictors.size_kb(r.model_id)) < 1e-6)
            check(f"{r.model_id} latency > 0", r.lat_median_ms > 0 and r.lat_mean_ms > 0)
            check(f"{r.model_id} throughput(1024) > throughput(1)", r.throughput_b1024_per_s > r.throughput_b1_per_s)
            check(f"{r.model_id} p99>=median", r.lat_p99_ms >= r.lat_median_ms)
            check(f"{r.model_id} cpu util in (0,150]", 0 < r.cpu_util_pct <= 150)


def notebook_checks():
    nbp = ROOT / "iot_ids.ipynb"
    if not nbp.exists():
        return
    nb = json.loads(nbp.read_text(encoding="utf-8"))
    heads = [l.lstrip("# ").strip() for c in nb["cells"] if c["cell_type"] == "markdown" for l in "".join(c["source"]).splitlines() if l.startswith("#")]
    order = ["Goal", "Dataset", "Inspection", "Cleaning", "Split", "Baseline training", "Evaluation", "Resource measurements", "Model reduction", "Final comparison"]
    pos = [next((i for i, h in enumerate(heads) if o.lower() in h.lower()), -1) for o in order]
    check("notebook sections in required order", all(p >= 0 for p in pos) and pos == sorted(pos), str(pos))
    tail = nb["cells"][-1]  # the optimization phase appends one pointer cell after the conclusion
    if "iot_ids_optimization" in "".join(tail["source"]):
        tail = nb["cells"][-2]
    check("notebook conclusion is last cell (before the phase-2 pointer)", "Conclusion" in "".join(tail["source"]))
    errs = [o for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    check("notebook has no error outputs", not errs)
    check("notebook code cells all executed", all(c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code"))
    text = "".join("".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", "")) + "".join(o.get("data", {}).get("text/html", ""))
                   for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []))
    for f in glob.glob(str(RES / "metrics_*.json")):
        m = jload(f)
        if m["model_id"] in {f"{n}_{t}" for n in ("dt", "rf", "xgb", "mlp") for t in ("binary", "multi")}:
            check(f"notebook shows {m['model_id']} macro_f1 = csv/json", f"{m['macro_f1']:.4f}" in text)
    for tag in ("docker", "native"):
        b = pd.read_csv(RES / f"baseline_comparison_{tag}.csv")
        for _, r in b.iterrows():
            check(f"notebook shows {tag} {r.model} binary size", f"{r.size_kb:.4f}" in text or f"{r.size_kb:.1f}" in text)


def leak_checks(force=False):
    """If any macro F1 > 0.99: shuffled-label control and single-feature models."""
    ms = [jload(f) for f in glob.glob(str(RES / "metrics_*.json"))]
    if not (force or any(m["macro_f1"] > 0.99 for m in ms)):
        return
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import f1_score
    Xtr, ytr = data.load("train", "multi"); Xte, yte = data.load("test", "multi")
    rng = np.random.default_rng(SEED)
    dt = DecisionTreeClassifier(max_depth=20, random_state=SEED).fit(Xtr, rng.permutation(ytr))
    f = f1_score(yte, dt.predict(Xte), average="macro")
    check("shuffled-label control collapses", f < 0.3, f"{f:.3f}")
    best = max(f1_score(yte, DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(Xtr[[c]], ytr).predict(Xte[[c]]), average="macro") for c in Xtr.columns)
    check("no single feature reaches 0.99", best < 0.99, f"{best:.3f}")


def optimization_checks():
    import hashlib, joblib
    O = RES / "optimization"; FIN = O / "final"
    if not (O / "preexisting_hashes.json").exists():
        return
    # --- no pre-existing artifact changed
    snap = jload(O / "preexisting_hashes.json")
    bad = [r for r, h in snap["files"].items() if not (ROOT / r).exists() or hashlib.sha256((ROOT / r).read_bytes()).hexdigest() != h]
    check("no pre-existing result/model/code file changed", not bad, str(bad[:5]))
    lg = (ROOT / "test_access.log").read_text().splitlines()
    check("test_access.log: original 80 lines unchanged (append-only)", lg[:len(snap["test_log_first_lines"])] == snap["test_log_first_lines"])
    cells = json.loads((ROOT / "iot_ids.ipynb").read_text(encoding="utf-8"))["cells"]
    check("iot_ids.ipynb original cells unchanged", ["".join(c["source"]) for c in cells[:len(snap["notebook_cell_sources"])]] == snap["notebook_cell_sources"])
    # --- budgets equal across methods
    runs = [jload(f) for f in glob.glob(str(O / "runs" / "*.json"))]
    bud = jload(O / "budgets.json")
    for k, b in bud.items():
        rs = [r for r in runs if f"{r['task']}_{r['model']}" == k]
        check(f"{k}: all 4 methods present", sorted(r["method"] for r in rs) == ["aco", "ga", "pso", "random"], str([r["method"] for r in rs]))
        check(f"{k}: equal budget across methods", all(r["budget"] == b for r in rs), str([(r["method"], r["budget"]) for r in rs]))
        # a run may stop short only when its search saturated on already-cached candidates (call guard 100 x budget); documented
        check(f"{k}: every method used the full budget (or stopped by cache saturation)", all(r["n_evals"] == b or (r["n_evals"] < b and r["n_calls"] > 100 * b) for r in rs), str([(r["method"], r["n_evals"], r["n_calls"]) for r in rs]))
        check(f"{k}: same pop and seed", len({(r["pop"], r["seed"]) for r in rs}) == 1)
        check(f"{k}: same first evaluated candidate", len({round(r["history"][0]["fitness"], 12) for r in rs}) == 1)
    # --- fitness code never reads test data
    for f in ("core.py", "optimizers.py", "run.py"):
        src = (ROOT / "optimization" / f).read_text(encoding="utf-8").lower()
        check(f"optimization/{f} never mentions the test split", "test" not in src, "contains 'test'")
    # --- test log: each final model once, and only after all optimizer runs finished
    ids = {Path(f).stem[len("metrics_"):] for f in glob.glob(str(FIN / "metrics_*.json"))}
    cnt = pd.Series([l.split("\t")[1] for l in lg]).value_counts()
    check("final models: each scored exactly once in test_access.log", all(cnt.get(i, 0) == 1 for i in ids), str([i for i in ids if cnt.get(i, 0) != 1][:5]))
    check("test log has no extra entries beyond final models", len(lg) - 80 == len(ids), f"{len(lg) - 80} vs {len(ids)}")
    last_run = max((Path(f).stat().st_mtime for f in glob.glob(str(O / "runs" / "*.json"))), default=0)
    first_t = min((pd.Timestamp(l.split("\t")[0]).timestamp() for l in lg[80:]), default=1e18)
    check("test scoring started only after all optimizer runs finished", first_t >= last_run - 2)
    # --- metrics recomputed from predictions; masks match models
    runmask = {(r["task"], r["model"], r["method"]): r["mask"] for r in runs}
    for mid in sorted(ids):
        m = jload(FIN / f"metrics_{mid}.json"); z = np.load(FIN / f"pred_{mid}.npz"); n = 2 if m["task"] == "binary" else len(CLASSES)
        r = compute(z["y_true"], z["y_pred"], n)
        check(f"{mid} metrics recompute", all(abs(r[k] - m[k]) < 1e-9 for k in ("macro_f1", "accuracy", "precision", "recall", "fpr")))
        check(f"{mid} macro_f1 independent", abs(own_macro_f1(z["y_true"], z["y_pred"], n) - m["macro_f1"]) < 1e-9)
        check(f"{mid} cm sums to test size", pd.read_csv(FIN / f"cm_{mid}.csv", index_col=0).to_numpy().sum() == len(z["y_true"]) == len(data.load("test", m["task"])[1]))
        meta = predictors.meta(mid); cfg = jload(FIN / f"config_{mid}.json")
        mask = runmask[(m["task"], m["model"], m["optimizer"])]
        check(f"{mid} saved cols == optimizer mask", meta["cols"] == cfg["cols"] == [i for i, b in enumerate(mask) if b])
        f0 = MOD / meta["files"][0]
        if meta["kind"] == "sklearn":
            nin = joblib.load(f0).n_features_in_
        elif meta["kind"] == "xgb":
            import xgboost as xgb; b = xgb.Booster(); b.load_model(f0); nin = b.num_features()
        else:
            import keras; nin = keras.models.load_model(f0, compile=False).input_shape[-1]
        check(f"{mid} model input width == #selected features", nin == len(meta["cols"]), f"{nin} vs {len(meta['cols'])}")
    for t in ("binary", "multi"):
        base = np.load(RES / f"pred_dt_{t}.npz")["y_true"]
        check(f"final {t} models use the same test labels as baselines", all(np.array_equal(base, np.load(FIN / f"pred_{i}.npz")["y_true"]) for i in ids if f"_{t}_" in i))
    # --- tables equal saved metrics; notebook equals tables
    from optimization import tables as T
    mt = T.metrics_table(); csv = pd.read_csv(O / "optimization_metrics.csv")
    check("optimization_metrics.csv equals metrics json files", len(mt) == len(csv) and np.allclose(mt[T.COLS].to_numpy(), csv[T.COLS].to_numpy(), atol=1e-12))
    check("optimization_metrics.csv has 4 models x 2 tasks x (baseline+4)", len(csv) == 40, str(len(csv)))
    nbp = ROOT / "iot_ids_optimization.ipynb"
    check("optimization notebook exists", nbp.exists())
    if nbp.exists():
        nb = json.loads(nbp.read_text(encoding="utf-8"))
        errs = [o for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
        check("optimization notebook executed without errors", not errs and all(c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code"))
        first = next(("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown" and "Results at a glance" in "".join(c["source"])), "")
        check("optimization notebook has 'Results at a glance' as first section", bool(first) and nb["cells"].index(next(c for c in nb["cells"] if "Results at a glance" in "".join(c["source"]))) <= 1)
        text = "".join("".join(o.get("data", {}).get("text/html", "")) + "".join(o.get("data", {}).get("text/plain", "")) for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []))
        for _, r in csv.iterrows():
            check(f"notebook shows {r.model_id} {r.optimizer} macro_f1 and fpr", f"{r.macro_f1:.4f}" in text and f"{r.fpr:.4f}" in text)
    if (O / "benchmark_docker.csv").exists():
        b = pd.read_csv(O / "benchmark_docker.csv")
        check("benchmark covers every final seed-42 model", set(b.model_id) == {i for i in ids if not i.endswith(("_s43", "_s44"))})
        for _, r in b.iterrows():
            if r.get("oom") is True or r.get("failed") is True:
                continue
            check(f"{r.model_id} bench size matches file", abs(r.size_kb - predictors.size_kb(r.model_id)) < 1e-6)


if __name__ == "__main__":
    optimization_checks()
    split_checks(); test_log_checks(); metric_checks(); model_size_checks(); notebook_checks(); leak_checks("--leak" in sys.argv)
    print(f"verify: {PASSES} passed, {len(FAILS)} failed")
    sys.exit(1 if FAILS else 0)
