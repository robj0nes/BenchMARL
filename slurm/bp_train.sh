#!/bin/bash
#SBATCH --nodes=1
##SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --mem=50000M
#SBATCH --partition gpu
#SBATCH --time-min=7200
#SBATCH --output ./outputs/dispersion2x.stdout
#SBATCH --error ./outputs/dispersion2x.stderr
#SBATCH --job-name=benchmarl_disp
#SBATCH --account=semt031264

cd "${SLURM_SUBMIT_DIR}"/..
nvidia-smi
nvidia-smi mig -lgip
MODEL='model=layers/gnn'
EXP='experiment.train_device="cuda" experiment.sampling_device="cuda"'
python benchmarl/run.py algorithm=mappo task=vmas/joint_passage "${MODEL}" "${EXP}"
# python benchmarl/run.py algorithm=mappo task=vmas/dispersion experiment.render=False
# python train/train_multi_give_way.py