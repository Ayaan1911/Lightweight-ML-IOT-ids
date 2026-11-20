"""Hash every pre-existing artifact (before the optimization phase) so verify.py can prove none changed."""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SKIP = {"verify.py", "README.md", "REPORT.md", "PROGRESS.md", "CHECKLIST.md", "DECISIONS.md", "iot_ids.ipynb", "test_access.log"}  # extended on purpose
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__ == "__main__":
    out = ROOT / "results" / "optimization" / "preexisting_hashes.json"
    if out.exists(): sys.exit("snapshot already exists")
    files = {}
    for p in ROOT.rglob("*"):
        r = p.relative_to(ROOT).as_posix()
        if p.is_file() and not r.startswith((".venv/", "results/optimization/", "optimization/", "__pycache__")) and "__pycache__" not in r and r not in SKIP and not r.startswith("MERGED_CSV/"):
            files[r] = h(p)
    log = (ROOT / "test_access.log").read_text().splitlines()
    nb = json.loads((ROOT / "iot_ids.ipynb").read_text(encoding="utf-8"))["cells"]
    out.write_text(json.dumps({"files": files, "test_log_first_lines": log, "notebook_cell_sources": ["".join(c["source"]) for c in nb]}, indent=1))
    print(len(files), "files hashed;", len(log), "log lines;", len(nb), "notebook cells")
