#!/bin/bash
#SBATCH --account=ebf11_c
##SBATCH --partition=burst
##SBATCH --qos=burst2x
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=8192
#SBATCH --time=48:00:00
#SBATCH --job-name=sdo_2014
#SBATCH --chdir=/storage/home/mlp95/work/sdo-clv-pipeline
#SBATCH --output=/storage/home/mlp95/work/logs/sdo_2014.%j.out

echo "About to start: $SLURM_JOB_NAME"
date
echo "Job id: $SLURM_JOBID"
echo "About to change into $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

echo "About to activate conda environment"
source /storage/group/ebf11/default/software/anaconda3/bin/activate
conda activate solar
echo "Environment activated"

echo "About to start Python"
python /storage/home/mlp95/work/sdo-clv-pipeline/scripts/run_pipe.py --globexp "*2014*"
# uv run scripts/run_pipe.py --globexp "*2014*"
echo "Python exited"
date
