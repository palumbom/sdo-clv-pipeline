#!/bin/bash
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks=64
##SBATCH --mem-per-cpu=8192
#SBATCH --time=72:00:00
#SBATCH --job-name=sdo_all
#SBATCH --chdir=/mnt/home/mpalumbo/work/savannah/sdo-clv-pipeline
#SBATCH --output=/mnt/home/mpalumbo/work/logs/sdo_all.%j.out

echo "About to start: $SLURM_JOB_NAME"
date
echo "Job id: $SLURM_JOBID"
echo "About to change into $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

echo "About to start Python"
uv run scripts/run_pipe.py --globexp ""
echo "Python exited"
date
