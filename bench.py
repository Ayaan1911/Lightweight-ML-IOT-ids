"""Benchmark ONE model in this fresh process.  python bench.py <model_id> [--pin] [--out file]
One thread everywhere, 20k-row saved input, warm-up. Run via run_bench.py (one subprocess per model)."""
import os, sys, json, time, argparse
os.environ.update({k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS")})
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3"); os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np, psutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
N_LAT, N_WARM, BATCH, REPS = 10_000, 200, 1024, 25


def peak_rss(proc):
    mi = proc.memory_info()
    return getattr(mi, "peak_wset", None) or __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mid"); ap.add_argument("--pin", action="store_true"); ap.add_argument("--out", default=None)
    ap.add_argument("--models", default=None, help="models dir override"); ap.add_argument("--inp", default=str(ROOT / "results" / "bench_input.npy"))
    a = ap.parse_args()
    proc = psutil.Process()
    if a.pin:
        proc.cpu_affinity([2])  # pin to one logical CPU (a P-core on the i5-12500H)
    from src import predictors
    if a.models:
        predictors.MOD = Path(a.models)
    X = np.load(a.inp).astype(np.float32)
    rss_before = proc.memory_info().rss
    predict = predictors.load(a.mid, threads=1)
    for _ in range(3):
        predict(X[:BATCH])
    rss_loaded = proc.memory_info().rss
    for i in range(N_WARM):
        predict(X[i:i + 1])
    out = {"model_id": a.mid, "size_kb": predictors.size_kb(a.mid), "pinned": a.pin, "env": "docker" if os.path.exists("/.dockerenv") else "native, not Docker-limited"}
    # batch-1 latency + cpu utilisation
    lat = np.empty(N_LAT); pred_peak = rss_loaded
    c0, w0 = sum(proc.cpu_times()[:2]), time.perf_counter()
    for i in range(N_LAT):
        r = X[i % len(X): i % len(X) + 1]
        t = time.perf_counter_ns(); predict(r); lat[i] = (time.perf_counter_ns() - t) / 1e6
        if i % 500 == 0:
            pred_peak = max(pred_peak, proc.memory_info().rss)
    c1, w1 = sum(proc.cpu_times()[:2]), time.perf_counter()
    cpu_b1 = (c1 - c0) / (w1 - w0); cpu_tot, wall_tot = c1 - c0, w1 - w0
    # batch-1024 throughput
    thr = []
    c0, w0 = sum(proc.cpu_times()[:2]), time.perf_counter()
    k = 0
    while k < REPS or time.perf_counter() - w0 < 2.0:  # >=25 repeats AND >=2 s so CPU-time ticks are meaningful
        s = (k * BATCH) % (len(X) - BATCH); k += 1
        t = time.perf_counter(); predict(X[s:s + BATCH]); thr.append(BATCH / (time.perf_counter() - t)); pred_peak = max(pred_peak, proc.memory_info().rss)
    c1, w1 = sum(proc.cpu_times()[:2]), time.perf_counter()
    cpu_b1024 = (c1 - c0) / (w1 - w0); cpu_all = (cpu_tot + c1 - c0) / (wall_tot + w1 - w0)
    pk = peak_rss(proc)
    out.update({
        "lat_mean_ms": float(lat.mean()), "lat_median_ms": float(np.median(lat)), "lat_p95_ms": float(np.percentile(lat, 95)),
        "lat_p99_ms": float(np.percentile(lat, 99)), "throughput_b1024_per_s": float(np.median(thr)), "n_reps_b1024": len(thr),
        "cpu_util_pct": 100 * cpu_all, "cpu_util_b1_pct": 100 * cpu_b1, "cpu_util_b1024_pct": 100 * cpu_b1024,
        "peak_rss_mb": pk / 2**20, "rss_loaded_mb": rss_loaded / 2**20, "rss_before_load_mb": rss_before / 2**20,
        "peak_rss_during_pred_mb": pred_peak / 2**20, "peak_rss_increase_mb": (pred_peak - rss_loaded) / 2**20,
        "throughput_b1_per_s": float(1000 / lat.mean()),
    })
    s = json.dumps(out)
    if a.out:
        Path(a.out).write_text(s)
    print(s)


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        print(json.dumps({"model_id": sys.argv[1], "oom": True}))
        sys.exit(3)
