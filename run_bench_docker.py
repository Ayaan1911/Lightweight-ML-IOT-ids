"""One `docker run --cpus=1 --memory=1g` per model; OOM/crash recorded as a result.  python run_bench_docker.py <ids...>"""
import sys, json, subprocess, os
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ["PATH"] += r";C:\Users\ASUS\AppData\Local\Programs\DockerDesktop\resources\bin"
out = ROOT / "results" / "bench_docker"; out.mkdir(exist_ok=True)
rows = []
for mid in sys.argv[1:]:
    f = out / f"{mid}.json"
    if not f.exists():
        cmd = ["docker", "run", "--rm", "--cpus=1", "--memory=1g", "--memory-swap=1g",
               "-v", f"{ROOT / 'models'}:/app/models:ro", "-v", f"{ROOT / 'results' / 'bench_input.npy'}:/app/results/bench_input.npy:ro",
               "-v", f"{out}:/out", "iot-bench", "python", "bench.py", mid, "--out", f"/out/{mid}.json"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not f.exists():
            oom = r.returncode in (137, -9) or "Killed" in r.stderr
            f.write_text(json.dumps({"model_id": mid, "failed": True, "oom": bool(oom), "returncode": r.returncode, "stderr": r.stderr[-300:]}))
        print(mid, r.returncode, flush=True)
    rows.append(json.loads(f.read_text()))
pd.DataFrame(rows).to_csv(ROOT / "results" / "benchmark_docker.csv", index=False)
