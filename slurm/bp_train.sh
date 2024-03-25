#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
##SBATCH --gpus-per-node=1
#SBATCH --mem=50000M
#SBATCH --partition gpu
#SBATCH --time-min=7200
#SBATCH --output ./outputs/paint_cutsomCritic_maddpg_noShare_2a-2g.stdout
#SBATCH --error ./outputs/paint_cutsomCritic_maddpg_noShare_2a-2g.stderr
#SBATCH --job-name=benchmarl_paint
#SBATCH --account=semt031264

cd "${SLURM_SUBMIT_DIR}"/..
nvidia-smi
nvidia-smi mig -lgip

CONFIG="algorithm=maddpg \
        algorithm.share_param_critic=False \
        experiment=base_experiment \
        experiment.sampling_device=cuda \
        experiment.train_device=cuda \
        experiment.share_policy_params=False \
        experiment.lr=0.0005 \
        task=vmas/painting \
        task.n_goals=2 \
        task.n_agents=2 \
        task.pos_shaping=True \
        model=layers/gnn \
        model@critic_model=critic_seq"

python benchmarl/run.py ${CONFIG}
