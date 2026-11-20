"""Run the 4 methods of each (task, model) as parallel processes, combos sequentially. Resumable.
python -m optimization.orchestrate probe | run"""
import sys, json, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY, OUT = sys.executable, ROOT / "results" / "optimization"
COMBOS = [(t, m) for t in ("binary", "multi") for m in ("dt", "rf", "xgb", "mlp")]
THREADS = 4
(OUT / "logs").mkdir(exist_ok=True)


def launch(t, m, jobs):
    ps = []
    for j in jobs:
        a = [PY, "-m", "optimization.run", t, m, j[0], str(j[1]), str(THREADS)] + ([str(j[2])] if len(j) > 2 else [])
        ps.append(subprocess.Popen(a, cwd=ROOT, stdout=open(OUT / "logs" / f"{t}_{m}_{j[0]}.log", "a"), stderr=subprocess.STDOUT))
    for p in ps: p.wait()


if __name__ == "__main__":
    if sys.argv[1] == "probe":
        for t, m in COMBOS:
            if not (OUT / "probe" / f"{t}_{m}_3.json").exists():
                launch(t, m, [("probe", 0, w) for w in range(4)])
    else:
        B = json.loads((OUT / "budgets.json").read_text())
        for t, m in COMBOS:
            todo = [(me, B[f"{t}_{m}"]) for me in ("ga", "pso", "aco", "random") if not (OUT / "runs" / f"{t}_{m}_{me}.json").exists()]
            if todo: launch(t, m, todo)
