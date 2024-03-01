#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
##SBATCH --gpus-per-node=1
#SBATCH --mem=50000M
#SBATCH --partition gpu
#SBATCH --time-min=7200
#SBATCH --output ./outputs/paint_maddpg_no_share.stdout
#SBATCH --error ./outputs/paint_maddpg_no_share.stderr
#SBATCH --job-name=benchmarl_gw
#SBATCH --account=semt031264

cd "${SLURM_SUBMIT_DIR}"/..
nvidia-smi
nvidia-smi mig -lgip
# python benchmarl/run.py algorithm=mappo experiment.share_policy_params=False experiment.lr=0.005 task=vmas/painting task.n_goals=4 model=layers/gnn
python benchmarl/run.py algorithm=maddpg experiment.share_policy_params=False experiment.lr=0.005 task=vmas/painting task.n_goals=4 model=layers/gnn
# python benchmarl/run.py algorithm=maddpg experiment.share_policy_params=False experiment.lr=0.005 task=vmas/painting task.n_goals=4 model=layers/gnn +task.position_shaping_factor=5.0

