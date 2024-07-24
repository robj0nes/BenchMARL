#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
##SBATCH --gpus-per-node=1
#SBATCH --mem=50000M
#SBATCH --partition gpu
#SBATCH --time-min=7200
#SBATCH --output ./outputs/simple_paint.stdout
#SBATCH --error ./outputs/simple_paint.stderr
#SBATCH --job-name=simple_paint
#SBATCH --account=semt031264

cd "${SLURM_SUBMIT_DIR}"/..

python benchmarl/experiment.py 
