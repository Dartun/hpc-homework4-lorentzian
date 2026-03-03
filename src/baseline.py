import argparse
import os
import numpy as np

# Use non-interactive backend so plotting never crashes on Windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def lorentzian_pdf(x):
    return 1.0 / (np.pi * (1.0 + x * x))


def lorentzian_cdf(x):
    return 0.5 + (1.0 / np.pi) * np.arctan(x)


def sample_lorentzian(n, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.random(n)            # Uniform(0,1)
    x = 1.0 / np.tan(np.pi * u)  # cot(pi*u) — correct Lorentzian/Cauchy
    return x


def make_histogram(x, bins=100, xmin=-10.0, xmax=10.0):
    counts, edges = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts, edges


def quantile_check(x):
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    print(f"Quantiles: Q1={q1:.4f}, median={med:.4f}, Q3={q3:.4f} (target ~ -1, 0, +1)")


def cdf_uniform_check(x):
    u1 = lorentzian_cdf(x)
    # Uniform(0,1): mean=0.5, var=1/12 ≈ 0.08333
    print(f"CDF->Uniform: mean={u1.mean():.5f} (target 0.5)")
    print(f"CDF->Uniform: var ={u1.var():.5f} (target ~0.08333)")


def save_hist_plot(counts, edges, outpath):
    bw = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_pdf = counts / (counts.sum() * bw)

    plt.figure()
    plt.plot(centers, hist_pdf, label="Empirical (hist PDF)")
    plt.plot(centers, lorentzian_pdf(centers), label="Analytic PDF")
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    print("Saved plot ->", outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--xmin", type=float, default=-10.0)
    ap.add_argument("--xmax", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    x = sample_lorentzian(args.n, seed=args.seed)
    counts, edges = make_histogram(x, bins=args.bins, xmin=args.xmin, xmax=args.xmax)
    print("Histogram count sum =", counts.sum())

    if args.checks:
        quantile_check(x)
        cdf_uniform_check(x)

    if args.plot:
        os.makedirs("reports", exist_ok=True)
        save_hist_plot(counts, edges, os.path.join("reports", "fig_hist_vs_pdf.png"))


if __name__ == "__main__":
    main()