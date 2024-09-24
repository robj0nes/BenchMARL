
for SEED in {0..4}
do
  export SEED
  export OUTPUT="./outputs/rdm_dims/6A6G_maddpg_seed"${SEED}""
          
  sbatch -o "${OUTPUT}.stdout" -e "${OUTPUT}.stderr" ./bp_train.sh  
done 