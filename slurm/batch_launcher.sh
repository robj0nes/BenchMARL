export CONFIG='experiment=bcp4_experiment'
export MODEL='model=layers/gnn'

# for ALGO in mappo maddpg masac ippo iddpg iql isac qmix vdn
for ALGO in mappo
do
  for SHARE_PARAMS in True False
  do
    export EXP="experiment.share_policy_params="${SHARE_PARAMS}""
    # No critic in IQL, VDN and QMIX
    if [ "${ALGO}" == "vdn" ] || [ "${ALGO}" == "iql" ] || [ "${ALGO}" == "qmix" ]
    then
      export OUTPUT="./outputs/"${ALGO}"_sp"${SHARE_PARAMS}"_scFalse"
      export ALGO
      export ALGO_CONF=''
      sbatch -o "${OUTPUT}.stdout" -e "${OUTPUT}.stderr" ./bcp4_train.sh 
    #   echo python benchmarl/run.py "${CONFIG}" algorithm="${ALGO}" task=vmas/painting "${MODEL}" experiment.share_policy_params="${SHARE_PARAMS}"
    else
      for SHARE_CRITIC in True False
      do
        export OUTPUT="./outputs/"${ALGO}"_sp"${SHARE_PARAMS}"_sc"${SHARE_CRITIC}""
        export ALGO
        export ALGO_CONF="algorithm.share_param_critic="${SHARE_CRITIC}""
        sbatch -o "${OUTPUT}.stdout" -e "${OUTPUT}.stderr" ./bcp4_train.sh 
        # echo python benchmarl/run.py "${CONFIG}" algorithm="${ALGO}" task=vmas/painting "${MODEL}" experiment.share_policy_params="${SHARE_PARAMS}" algorithm.share_param_critic="${SHARE_CRITIC}"
      done
    fi 
  done    
done 