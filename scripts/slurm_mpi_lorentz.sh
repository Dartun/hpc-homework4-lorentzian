#!/bin/bash
#SBATCH -J hw4_lorentz_mpi
#SBATCH -A utdallas
#SBATCH -p cpu-preempt
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:05:00

module purge
module load gnu12/12.4.0
module load openmpi4/4.1.6

# Activate your MPI-capable venv (miniforge-based)
source /scratch/ganymede2/tda230002/hw4/hpc-homework4-lorentzian/.venv_mpi/bin/activate

# Avoid UCX locked-memory issues
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp

mpirun -np ${SLURM_NTASKS} /scratch/ganymede2/tda230002/hw4/hpc-homework4-lorentzian/.venv_mpi/bin/python \
  src/mpi_lorentz.py --n_total 2000000 --out results/mpi_hist_2e6.txt

