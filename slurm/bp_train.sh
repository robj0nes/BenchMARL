#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
##SBATCH --gpus-per-node=1
#SBATCH --mem=50000M
#SBATCH --partition gpu
#SBATCH --time-min=7200
#SBATCH --output ./outputs/gw_gnn_NOshare_500step.stdout
#SBATCH --error ./outputs/gw_gnn_NOshare_500step.stderr
#SBATCH --job-name=benchmarl_gw
#SBATCH --account=semt031264

cd "${SLURM_SUBMIT_DIR}"/..
nvidia-smi
nvidia-smi mig -lgip
python benchmarl/run.py algorithm=mappo experiment.share_policy_params=False task=vmas/give_way model=layers/gnn
# python benchmarl/run.py algorithm=mappo task=vmas/give_way model=layers/gnn
# python train/train_multi_give_way.py