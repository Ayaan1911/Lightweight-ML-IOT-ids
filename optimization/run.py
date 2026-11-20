"""One optimizer run (or timing probe).  python -m optimization.run <task> <model> <method|probe> <budget> <threads> [worker]"""
import os, sys, time
task, model, method, budget, threads = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS"):
    os.environ[v] = str(threads)
os.environ.update(TF_CPP_MIN_LOG_LEVEL="3", TF_ENABLE_ONEDNN_OPTS="0")
import numpy as np
from src.common import ROOT, jdump
from optimization import core, optimizers as O

OUT = ROOT / "results" / "optimization"
POP, SEED = 20, 1


def main():
    d = core.dim(model)
    fn = lambda x: core.fitness(model, task, x, threads)
    if method == "probe":  # time 6 random candidates after one warm-up
        wk = int(sys.argv[6]); rng = np.random.default_rng(100 + wk)
        core.data(task); ts = [fn(rng.random(d))[1]["eval_s"] for _ in range(7)][1:]
        (OUT / "probe").mkdir(exist_ok=True)
        jdump({"task": task, "model": model, "worker": wk, "eval_s": ts, "mean": float(np.mean(ts))}, OUT / "probe" / f"{task}_{model}_{wk}.json")
        return
    f = OUT / "runs" / f"{task}_{model}_{method}.json"
    if f.exists():
        return
    core.data(task)
    obj = O.Objective(fn, lambda x: core.key(model, x), budget)
    t0, c0 = time.time(), time.process_time()
    O.run(method, obj, d, POP, SEED)
    mask, hp = core.decode(model, obj.best_x)
    jdump({"task": task, "model": model, "method": method, "budget": budget, "pop": POP, "seed": SEED, "n_evals": len(obj.hist), "n_calls": obj.calls,
           "best_fitness": obj.best, "best_x": obj.best_x.tolist(), "mask": mask.astype(int).tolist(), "hp": hp, "threads": threads,
           "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0, "history": obj.hist}, f)
    print(task, model, method, "done", len(obj.hist), round(obj.best, 4), flush=True)


if __name__ == "__main__":
    main()
