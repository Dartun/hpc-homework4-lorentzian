# Sections 4–5 (Ganymede) — Benchmarking Notes

## Measurement protocol (Section 4)
- Timing: `time.perf_counter()` around the compute region
- Warmup: 1 warmup run before timed trials
- Trials: 5 trials per (method, p, bins) point; mean/std recorded
- Oversubscription control:
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
- Metadata captured per run: host, python version, method, n, p, bins, env vars
- Raw JSON outputs: `results/bench/*.json`

## Strong scaling (Section 5)
- Fixed n_total = 10,000,000
- p = 1,2,4,8,16,32,64
- bins = 10, 100, 1000
- Main method: `ppe` (ProcessPoolExecutor)
- Additional comparisons: `thread`, `mp`
- Plots: `reports/runtime_strong_*`, `reports/speedup_strong_*`, `reports/efficiency_strong_*`
- CSV summaries: `reports/summary/summary_strong_*.csv`

## Weak scaling (Section 5)
- n_per_worker = 2,000,000 (so n_total = p * n_per_worker)
- p = 1,2,4,8,16,32,64
- bins = 10, 100, 1000
- Plots: `reports/runtime_weak_*`, `reports/speedup_weak_*`, `reports/efficiency_weak_*`
- CSV summaries: `reports/summary/summary_weak_*.csv`

## Grad-only notes
- Amdahl serial fraction estimates printed by `analysis/plot_results.py`
- Largest completed run (≤10 min): p=64, bins=100, n_total up to 1e8 (see `results/bench/*final.json`)
