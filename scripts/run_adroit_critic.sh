export D4RL_SUPPRESS_IMPORT_ERROR=1
# export CUDA_VISIBLE_DEVICES=0
# export WANDB_DISABLED=True

algo=$1
seed=$2
env=$3
eval_dataset_fp=$4
pretrained_agent_fp=$5
alignment_episodes=$6
max_epochs=$7
critic_utd_ratio=$8

# env=pen-binary-v0
# env=door-binary-v0
# env=relocate-binary-v0

if [ "$env" = "pen-binary-v0" ]; then
    max_online_env_steps=3e5
elif [ "$env" = "door-binary-v0" ] || [ "$env" = "relocate-binary-v0" ]; then
    max_online_env_steps=1e6
fi



logging_output_dir=/home/suj/Desktop/projects/Cal-QL/experiment_result
logging_prefix="$algo"

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m JaxCQL.conservative_sac_policy_eval_main \
    --env=$env \
    --seed=$seed \
    --eval_dataset_fp=${eval_dataset_fp} \
    --pretrained_agent_fp=${pretrained_agent_fp} \
    --alignment_episodes=${alignment_episodes} \
    --max_epochs=${max_epochs} \
    --critic_utd_ratio=${critic_utd_ratio} \
    --logging.online \
    --logging.prefix=${logging_prefix} \
    --logging.output_dir=${logging_output_dir} \
    --logging.project=$env-${alignment_episodes}episodes-${max_epochs}epochs-critic_utd_ratio${critic_utd_ratio} \
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
    --save_model=True
