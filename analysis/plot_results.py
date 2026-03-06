import argparse, glob, json, os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_jsons(pattern):
    files = sorted(glob.glob(pattern))
    rows = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        rows.append(d)
    return rows

def summarize(rows):
    # returns dict bins -> list of (p, mean, std)
    by_bins = {}
    for d in rows:
        bins = int(d["bins"])
        p = int(d["p"])
        by_bins.setdefault(bins, []).append((p, d["time_mean_sec"], d["time_std_sec"]))
    for b in by_bins:
        by_bins[b].sort(key=lambda x: x[0])
    return by_bins

def write_csv(method, mode, tag, bins, triples):
    os.makedirs("results", exist_ok=True)
    out = f"results/summary_{mode}_{method}_bins{bins}_{tag}.csv"
    lines = ["p,time_mean_sec,time_std_sec,speedup,efficiency"]
    t1 = triples[0][1]
    for p, mean, std in triples:
        sp = t1 / mean
        eff = sp / p
        lines.append(f"{p},{mean:.6f},{std:.6f},{sp:.4f},{eff:.4f}")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out

def plot_runtime(method, mode, tag, bins, triples):
    ps = [t[0] for t in triples]
    means = [t[1] for t in triples]
    stds = [t[2] for t in triples]

    plt.figure()
    plt.errorbar(ps, means, yerr=stds, marker="o", linestyle="-", capsize=3)
    plt.xlabel("p (workers)")
    plt.ylabel("Runtime (s)")
    plt.title(f"{method} {mode} scaling (bins={bins})")
    plt.tight_layout()

    os.makedirs("reports", exist_ok=True)
    out = f"reports/runtime_{mode}_{method}_bins{bins}_{tag}.png"
    plt.savefig(out, dpi=200)
    return out

def plot_speedup_eff(method, mode, tag, bins, triples):
    ps = [t[0] for t in triples]
    means = [t[1] for t in triples]
    t1 = means[0]
    speedup = [t1 / m for m in means]
    eff = [s / p for s, p in zip(speedup, ps)]

    os.makedirs("reports", exist_ok=True)

    plt.figure()
    plt.plot(ps, speedup, marker="o")
    plt.xlabel("p (workers)")
    plt.ylabel("Speedup S_p")
    plt.title(f"Speedup: {method} {mode} (bins={bins})")
    plt.tight_layout()
    out1 = f"reports/speedup_{mode}_{method}_bins{bins}_{tag}.png"
    plt.savefig(out1, dpi=200)

    plt.figure()
    plt.plot(ps, eff, marker="o")
    plt.xlabel("p (workers)")
    plt.ylabel("Efficiency E_p")
    plt.title(f"Efficiency: {method} {mode} (bins={bins})")
    plt.tight_layout()
    out2 = f"reports/efficiency_{mode}_{method}_bins{bins}_{tag}.png"
    plt.savefig(out2, dpi=200)

    return out1, out2

def estimate_serial_fraction(ps, times):
    # Amdahl: S(p)=1/(f + (1-f)/p) => f = (1/S - 1/p) / (1 - 1/p)
    t1 = times[0]
    f_est = []
    for p, tp in zip(ps[1:], times[1:]):
        S = t1 / tp
        f = (1.0 / S - 1.0 / p) / (1.0 - 1.0 / p)
        f_est.append(f)
    return float(np.mean(f_est)) if f_est else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--mode", choices=["strong","weak"], required=True)
    ap.add_argument("--tag", default="ganymede")
    args = ap.parse_args()

    pattern = f"results/bench/{args.method}_p*_n*_bins*_{args.tag}.json"
    rows = load_jsons(pattern)
    if not rows:
        raise SystemExit(f"No results found for pattern: {pattern}")

    by_bins = summarize(rows)
    for bins, triples in sorted(by_bins.items()):
        csv_path = write_csv(args.method, args.mode, args.tag, bins, triples)
        rt_path = plot_runtime(args.method, args.mode, args.tag, bins, triples)
        sp_path, ef_path = plot_speedup_eff(args.method, args.mode, args.tag, bins, triples)
        print("WROTE", csv_path)
        print("WROTE", rt_path)
        print("WROTE", sp_path)
        print("WROTE", ef_path)

        # Grad-only: serial fraction estimate for strong scaling
        if args.mode == "strong":
            ps = [t[0] for t in triples]
            times = [t[1] for t in triples]
            f = estimate_serial_fraction(ps, times)
            print(f"Estimated serial fraction (Amdahl) for bins={bins}: f ≈ {f:.4f}")

if __name__ == "__main__":
    main()
