export CONFIG='experiment=bp_experiment'
export AGENT_MODEL='model=layers/gnn'
export CRITIC_MODEL='model@critic_model=layers/deepsets'

# for ALGO in mappo maddpg masac ippo iddpg iql isac qmix vdn
for ALGO in maddpg
export ALGO
do
  for SEED in {0..4}
  do
    export SEED
    for SHARE_PARAMS in False
    do
      export EXP="experiment.share_policy_params="${SHARE_PARAMS}""
      # No critic in IQL, VDN and QMIX
      if [ "${ALGO}" == "vdn" ] || [ "${ALGO}" == "iql" ] || [ "${ALGO}" == "qmix" ]
      then
        export OUTPUT="./outputs/"${ALGO}"_sp"${SHARE_PARAMS}"_scFalse_seed"${SEED}"" 
        export ALGO_CONF=''
        # sbatch -o "${OUTPUT}.stdout" -e "${OUTPUT}.stderr" ./bp_train.sh 
      #   echo python benchmarl/run.py "${CONFIG}" algorithm="${ALGO}" task=vmas/painting "${MODEL}" experiment.share_policy_params="${SHARE_PARAMS}"
      else
        for SHARE_CRITIC in True
        do
          export OUTPUT="./outputs/"${ALGO}"_sp"${SHARE_PARAMS}"_sc"${SHARE_CRITIC}"_seed"${SEED}""
          export ALGO_CONF="algorithm.share_param_critic="${SHARE_CRITIC}""
          sbatch -o "${OUTPUT}.stdout" -e "${OUTPUT}.stderr" ./bp_train.sh 
          # echo benchmarl/run.py seed="${SEED}" "${CONFIG}" "${EXP}" "${AGENT_MODEL}" "${CRITIC_MODEL}" algorithm="${ALGO}" "${ALGO_CONF}" task=vmas/painting 
          # echo python benchmarl/run.py "${CONFIG}" algorithm="${ALGO}" task=vmas/painting "${MODEL}" experiment.share_policy_params="${SHARE_PARAMS}" algorithm.share_param_critic="${SHARE_CRITIC}"
        done
      fi 
    done 
  done     
done 