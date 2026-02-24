#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J bhsuper_mc
#SBATCH --mail-user=clwelch@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -A m3166
#SBATCH -t 2:0:0
#SBATCH --array=0-35

# 1. Threading settings for Python/Numpy 
# These prevent internal Numpy loops from fighting with Bilby's multiprocessing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 2. Setup environment
mkdir -p logs
module load python/3.11
module load conda
conda activate BHsuper

# 3. The Application Run
# We use -c 2 to account for hyperthreading (256 logical / 128 physical)
# This ensures each of your 128 'npool' workers gets its own physical core.
srun -n 1 -c 256 --cpu-bind=none python /pscratch/sd/c/clwelch/BH-superradiance/src/cluster_event_sampling.py $SLURM_ARRAY_TASK_ID