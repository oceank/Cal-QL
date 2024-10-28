export D4RL_SUPPRESS_IMPORT_ERROR=1
# export CUDA_VISIBLE_DEVICES=0
# export WANDB_DISABLED=True

algo=$1
seed=$2
env=$3
# env=pen-binary-v0
# env=door-binary-v0
# env=relocate-binary-v0

if [ "$env" = "pen-binary-v0" ]; then
    max_online_env_steps=3e5
elif [ "$env" = "door-binary-v0" ] || [ "$env" = "relocate-binary-v0" ]; then
    max_online_env_steps=1e6
fi

enable_calql=False
if [[ "$algo" == "CQL" ]]; then
   enable_calql=False
elif [[ "$algo" == "Cal-QL" ]]; then
   enable_calql=True
else
   echo "Unsupported algo $algo"
   exit 1
fi

logging_output_dir=/home/suj/Desktop/projects/Cal-QL/experiment_result
logging_prefix="$algo"

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m JaxCQL.conservative_sac_aligned_main \
    --env=$env \
    --seed=$seed \
    --logging.online \
    --logging.prefix=${logging_prefix} \
    --logging.output_dir=${logging_output_dir} \
    --logging.project=$algo-$env \
    --cql_min_q_weight=1.0 \
    --policy_arch=512-512 \
    --qf_arch=512-512-512 \
    --offline_eval_every_n_epoch=2 \
    --online_eval_every_n_env_steps=2000 \
    --eval_n_trajs=20 \
    --n_train_step_per_epoch_offline=1000 \
    --n_pretrain_epochs=20 \
    --max_online_env_steps=$max_online_env_steps \
    --mixing_ratio=0.5 \
    --reward_scale=10.0 \
    --reward_bias=5.0 \
    --enable_calql=${enable_calql} \
    --save_model=True
