import argparse, subprocess, sys

def run(cmd):
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--mode", choices=["strong","weak"], required=True)
    ap.add_argument("--n", type=int, required=True, help="strong: fixed n; weak: n_per_worker")
    ap.add_argument("--bins_list", type=str, default="10,100,1000")
    ap.add_argument("--p_list", type=str, required=True)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--scheduler", type=str, default="processes")
    ap.add_argument("--tag", type=str, default="ganymede")
    args = ap.parse_args()

    bins_list = [int(x) for x in args.bins_list.split(",")]
    p_list = [int(x) for x in args.p_list.split(",")]

    for bins in bins_list:
        for p in p_list:
            if args.mode == "strong":
                n_total = args.n
            else:
                n_total = args.n * p

            cmd = [
                sys.executable, "analysis/bench_run.py",
                "--method", args.method,
                "--n", str(n_total),
                "--p", str(p),
                "--bins", str(bins),
                "--trials", str(args.trials),
                "--warmup", str(args.warmup),
                "--scheduler", args.scheduler,
                "--tag", args.tag
            ]
            run(cmd)

if __name__ == "__main__":
    main()
