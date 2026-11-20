"""Run bench.py once per model, each in a fresh process.  python run_bench.py <tag> <id> [<id> ...]
Docker mode: python run_bench.py --docker <tag> <ids...> (needs Docker; see run_docker.sh)."""
import sys, subprocess, json, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent
args = sys.argv[1:]
docker = args[0] == "--docker"
if docker: args = args[1:]
tag, ids = args[0], args[1:]
out = ROOT / "results" / "bench"; out.mkdir(exist_ok=True)
rows = []
for mid in ids:
    f = out / f"{mid}.json"
    if f.exists():
        rows.append(json.loads(f.read_text())); continue
    cmd = [sys.executable, "bench.py", mid, "--out", str(f)] + ([] if docker else ["--pin"])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if f.exists():
        rows.append(json.loads(f.read_text())); print(mid, "ok", flush=True)
    else:
        rows.append({"model_id": mid, "failed": True, "stderr": r.stderr[-300:], "returncode": r.returncode}); print(mid, "FAILED", r.returncode, r.stderr[-200:], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / f"benchmark_{tag}.csv", index=False)
