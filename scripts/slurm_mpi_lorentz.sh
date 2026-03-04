#!/bin/bash
#SBATCH -J hw4_lorentz_mpi
#SBATCH -A your_account
#SBATCH -p compute
#SBATCH -N 2
#SBATCH -n 16
#SBATCH -t 00:10:00
#SBATCH --exclusive

module purge
module load python
module load mpi

# Avoid oversubscription with threaded BLAS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Activate venv if you use one on the cluster
# source .venv/bin/activate

srun -n 16 python src/mpi_lorentz.py --n_total 10000000 --bins 100 --xmin -10 --xmax 10 --seed 42 --out results/mpi_lorentzian_histogram.txt