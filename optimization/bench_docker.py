"""Docker benchmark (--cpus=1 --memory=1g, same image/bench.py as the baselines) for the optimized models.
OOM/crash is recorded as a result.  python -m optimization.bench_docker"""
import sys, json, subprocess, os, glob
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ["PATH"] += r";C:\Users\ASUS\AppData\Local\Programs\DockerDesktop\resources\bin"
OUT = ROOT / "results" / "optimization" / "bench_docker"; OUT.mkdir(exist_ok=True)
ids = sorted(Path(f).stem[len("metrics_"):] for f in glob.glob(str(ROOT / "results/optimization/final/metrics_*.json"))
             if "_s43" not in f and "_s44" not in f)
rows = []
for mid in ids:
    f = OUT / f"{mid}.json"
    if not f.exists():
        cmd = ["docker", "run", "--rm", "--cpus=1", "--memory=1g", "--memory-swap=1g",
               "-v", f"{ROOT / 'models'}:/app/models:ro", "-v", f"{ROOT / 'results' / 'bench_input.npy'}:/app/results/bench_input.npy:ro",
               "-v", f"{OUT}:/out", "iot-bench", "python", "bench.py", mid, "--out", f"/out/{mid}.json"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not f.exists():
            f.write_text(json.dumps({"model_id": mid, "failed": True, "oom": bool(r.returncode in (137, -9) or "Killed" in r.stderr),
                                     "returncode": r.returncode, "stderr": r.stderr[-300:]}))
        print(mid, r.returncode, flush=True)
    rows.append(json.loads(f.read_text()))
pd.DataFrame(rows).to_csv(ROOT / "results" / "optimization" / "benchmark_docker.csv", index=False)
