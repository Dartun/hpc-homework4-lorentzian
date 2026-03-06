import argparse, json, os, platform, socket, time
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from lorentzian_core import lorentzian_histogram
from thread_lorentz import run_threaded
from mp_lorentz import run_multiproc
from ppe_lorentz import run_ppe
from async_lorentz import run_async

def _try_import(name):
    try:
        return __import__(name, fromlist=["*"])
    except Exception:
        return None

dask_mod = _try_import("dask_lorentz")
joblib_mod = _try_import("joblib_lorentz")
mpire_mod = _try_import("mpire_lorentz")
numba_mod = _try_import("numba_lorentz")

def run_method(method, n, p, bins, xmin, xmax, seed, scheduler="processes"):
    if method == "baseline":
        return lorentzian_histogram(n, bins=bins, xmin=xmin, xmax=xmax, seed=seed)

    if method == "thread":
        return run_threaded(n, n_threads=p, bins=bins, xmin=xmin, xmax=xmax, seed=seed)

    if method == "mp":
        return run_multiproc(n, n_cores=p, bins=bins, xmin=xmin, xmax=xmax, seed=seed)

    if method == "ppe":
        return run_ppe(n, max_workers=p, bins=bins, xmin=xmin, xmax=xmax, seed=seed)

    if method == "async":
        return run_async(n, n_tasks=p, bins=bins, xmin=xmin, xmax=xmax, seed=seed)

    if method == "dask":
        if dask_mod is None:
            raise RuntimeError("dask not installed. pip install dask")
        return dask_mod.run_dask(n, n_tasks=p, bins=bins, xmin=xmin, xmax=xmax, seed=seed, scheduler=scheduler)

    if method == "joblib":
        if joblib_mod is None:
            raise RuntimeError("joblib not installed. pip install joblib")
        return joblib_mod.run_joblib(n, n_jobs=p, bins=bins, xmin=xmin, xmax=xmax, seed=seed)

    if method == "mpire":
        if mpire_mod is None:
            raise RuntimeError("mpire not installed. pip install mpire")
        return mpire_mod.run_mpire(n, n_jobs=p, bins=bins, xmin=xmin, xmax=xmax, seed=seed)

    if method == "numba":
        if numba_mod is None:
            raise RuntimeError("numba not installed. pip install numba")
        return numba_mod.run_numba(n, bins=bins, xmin=xmin, xmax=xmax, seed=seed)

    raise ValueError(f"Unknown method: {method}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["baseline","thread","mp","ppe","async","dask","joblib","mpire","numba"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--xmin", type=float, default=-10.0)
    ap.add_argument("--xmax", type=float, default=10.0)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--scheduler", type=str, default="processes", help="for dask: threads|processes")
    ap.add_argument("--tag", type=str, default="", help="extra tag in filename")
    args = ap.parse_args()

    os.makedirs("results/bench", exist_ok=True)

    meta = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "method": args.method,
        "n": args.n,
        "p": args.p,
        "bins": args.bins,
        "range": [args.xmin, args.xmax],
        "trials": args.trials,
        "warmup": args.warmup,
        "seed": args.seed,
        "scheduler": args.scheduler,
        "env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
    }

    # Warmup
    for i in range(args.warmup):
        _ = run_method(
            args.method,
            max(50_000, args.n // 50),
            max(1, args.p),
            args.bins,
            args.xmin,
            args.xmax,
            args.seed + i,
            scheduler=args.scheduler,
        )

    times = []
    sums = []
    for t in range(args.trials):
        t0 = time.perf_counter()
        counts = run_method(
            args.method, args.n, args.p, args.bins, args.xmin, args.xmax,
            args.seed + 1000 * t, scheduler=args.scheduler
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)
        sums.append(int(np.sum(counts)))

    out = dict(meta)
    out["times_sec"] = times
    out["time_mean_sec"] = float(np.mean(times))
    out["time_std_sec"] = float(np.std(times, ddof=1)) if args.trials > 1 else 0.0
    out["hist_sum"] = sums

    tag = f"_{args.tag}" if args.tag else ""
    fname = f"results/bench/{args.method}_p{args.p}_n{args.n}_bins{args.bins}{tag}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("WROTE", fname)
    print("mean_sec", out["time_mean_sec"], "std_sec", out["time_std_sec"])

if __name__ == "__main__":
    main()
