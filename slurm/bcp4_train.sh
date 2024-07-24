#!/bin/bash -login
#SBATCH --job-name benchmarl
#SBATCH --nodes 1
#SBATCH --gpus=1
#SBATCH --partition gpu
#SBATCH --mem=50000M
#SBATCH --account=semt031264

module load cuda
conda activate benchmarl

cd "${SLURM_SUBMIT_DIR}"/..

echo Running on host `hostname`
nvidia-smi
# nvidia-smi mig -lgip
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOB_ID
echo This job runs on the following machines:
echo `echo $SLURM_JOB_NODELIST | uniq`

# These values are exported from the batch_launcher script
echo Config = "${CONFIG}"
echo Model = "${MODEL}"
echo Algorithm = "${ALGO}"
echo Algorithm Config = "${ALGO_CONF}"
echo Experiment Config = "${EXP}"

python benchmarl/run.py experiment=bcp4_experiment model=layers/gnn algorithm=maddpg task=vmas/painting
# python benchmarl/run.py "${CONFIG}" "${EXP}" "${MODEL}" algorithm="${ALGO}" "${ALGO_CONF}" task=vmas/painting 
