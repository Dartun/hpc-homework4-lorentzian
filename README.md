# hpc-homework4-lorentzian
rm -f README.md
cat > README.md << 'EOF'
# HPC Homework 4 — Lorentzian Sampling + Parallel Scaling

This project generates random samples from a Lorentzian (Cauchy) distribution using inverse transform sampling and benchmarks multiple parallel Python approaches for a CPU-bound workload.

## Quick start (baseline)
```bash
python -m venv .venv
# Git Bash on Windows:
source .venv/Scripts/activate
pip install -r requirements.txt
python src/baseline.py --n 2000000 --plot
