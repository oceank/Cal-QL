 
import numpy as np
import gym
import d4rl
import json
from datetime import datetime

import absl.app
import absl.flags
from mlxu.logging import load_pickle
from ml_collections.config_dict.config_dict import ConfigDict

from .conservative_sac import ConservativeSAC
from .replay_buffer import subsample_batch, concatenate_batches, load_dataset_h5py, get_rb_dataset_with_mc_calculation
from .jax_utils import batch_to_jax
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import TrajSampler
from .utils import (
    Timer, define_flags_with_default, set_random_seed,
    get_user_flags, prefix_metrics, WandBLogger
)
from viskit.logging import logger, setup_logger
from .replay_buffer import ReplayBuffer

from functools import singledispatch

@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


FLAGS_DEF = define_flags_with_default(
    env='antmaze-medium-diverse-v2',
    seed=42,
    save_model=False,
    offline_dataset_fp="",
    offline_dataset_use_buffer=True,
    offline_validation_budget=50000,
    eval_episodes_per_evaluation=3,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.99999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=True,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    # Total grad_steps of offline pretrain will be (n_train_step_per_epoch_offline * n_pretrain_epochs)
    n_train_step_per_epoch_offline=1000,
    n_pretrain_epochs=1000,
    offline_eval_every_n_epoch=10,

    max_online_env_steps=1e6,
    online_eval_every_n_env_steps=1000,

    eval_n_trajs=5,
    replay_buffer_size=1000000,
    mixing_ratio=-1.0,
    use_cql=True,
    online_use_cql=True,
    cql_min_q_weight=5.0,
    cql_min_q_weight_online=-1.0,
    enable_calql=True, # Turn on for Cal-QL

    n_online_traj_per_epoch=1,
    online_utd_ratio=1,

    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)

max_steps_per_episode_dict = {
    "kitchen-complete-v0": 280,
    "kitchen-mixed-v0": 280,
    "kitchen-new-v0": 280,
    "pen-expert-v0": 100,
    "door-expert-v0": 200,
    "hammer-expert-v0": 200,
    "Ant-v2": 1000,
    "Hopper-v2": 1000,
    "Walker2d-v2": 1000,
    "maze2d-umaze-v1": 300,
    "maze2d-medium-v1": 600,
    "maze2d-large-v1": 800,
}

def main(argv):
    FLAGS = absl.flags.FLAGS
    
    now = datetime.now()
    expr_time_str = now.strftime("%Y%m%d-%H%M%S")
    algo = "CalQL" if FLAGS.enable_calql else "CQL"
    offline_dataset_tag = "UseBuffer" if FLAGS.offline_dataset_use_buffer else "UseNewData"
    experiment_result_folder_name = f"ft_{algo}_{FLAGS.env}_seed{FLAGS.seed}_offlineDataset{offline_dataset_tag}_{expr_time_str}"
    expr_dir = f"{FLAGS.logging.output_dir}/{experiment_result_folder_name}"
    FLAGS.logging.output_dir = expr_dir
    
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    # save experiment configuration to file
    flags_dict = absl.flags.FLAGS.flags_by_module_dict()
    flags_dict = {k: {v.name: v.value for v in vs} for k, vs in flags_dict.items()}
    expr_config_filepath = f"{FLAGS.logging.output_dir}/expr_config.json"
    expr_config_dict = {k:(v.to_dict() if isinstance(v, ConfigDict) else v) for k, v in flags_dict.items()}
    with open(expr_config_filepath, "w") as f:
        json.dump(expr_config_dict, f, indent=4, default=to_serializable)

    # load offline dataset
    offline_dataset, _ = load_dataset_h5py(FLAGS.offline_dataset_fp)
    dataset = get_rb_dataset_with_mc_calculation(FLAGS.env, FLAGS.reward_scale, FLAGS.reward_bias, FLAGS.clip_action, FLAGS.cql.discount, offline_dataset)
    use_goal = False

    # offline RL validation setup: similar to the setup for online RL validation
    offline_eval_episodes = int(FLAGS.offline_validation_budget / max_steps_per_episode_dict[FLAGS.env])
    offline_evals = offline_eval_episodes//FLAGS.eval_episodes_per_evaluation
    offline_eval_epochs = list(
        range(FLAGS.n_pretrain_epochs, 0, -1*(FLAGS.n_pretrain_epochs//(offline_evals-1)))
        )
    if (len(offline_eval_epochs) + 1) == offline_evals:
        # the agent at the step 0 definitely has a randomly initialized policy. So, do not evaluate here
        offline_eval_epochs.append(offline_eval_epochs[-1]//2) # the step at the middle of step 0 and the the currently earliest evaluation step


    assert dataset["next_observations"].shape == dataset["observations"].shape

    set_random_seed(FLAGS.seed)
    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, use_goal, gamma=FLAGS.cql.discount)
    train_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, use_goal, use_mc=True, gamma=FLAGS.cql.discount, reward_scale=FLAGS.reward_scale, reward_bias=FLAGS.reward_bias,)
    replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size)

    observation_dim = eval_sampler.env.observation_space.shape[0]
    action_dim = eval_sampler.env.action_space.shape[0]

    policy = TanhGaussianPolicy(
        observation_dim, action_dim, FLAGS.policy_arch, FLAGS.orthogonal_init,
        FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset
    )

    qf = FullyConnectedQFunction(observation_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init)

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = ConservativeSAC(FLAGS.cql, policy, qf)
    sampler_policy = SamplerPolicy(sac.policy, sac.train_params['policy'])

    # help save the best offline agent and its performance
    offline_eval_filepath = f"{FLAGS.logging.output_dir}/eval_ave_episode_return_offline{offline_dataset_tag}.txt"
    with open(offline_eval_filepath, "w") as f:
        f.write(f"Experiment Time\tEpoch\tReturn\n")
    offline_eval_file = open(offline_eval_filepath, "a")
    best_offline_agent_epoch = None
    best_offline_agent_perf = None
    best_offline_agent_filename = f"best_offline_agent_{offline_dataset_tag}.pkl"
    best_offline_agent_filepath = f"{FLAGS.logging.output_dir}/{best_offline_agent_filename}"

    # help log the evaluation of the fine-tuning process
    ft_eval_filepath = f"{FLAGS.logging.output_dir}/eval_ave_episode_return_ft.txt"
    with open(ft_eval_filepath, "w") as f:
        f.write(f"Experiment Time\tEpoch\tReturn\n")
    ft_eval_file = open(ft_eval_filepath, "a")

    viskit_metrics = {}
    n_train_step_per_epoch = FLAGS.n_train_step_per_epoch_offline
    cql_min_q_weight = FLAGS.cql_min_q_weight
    enable_calql=FLAGS.enable_calql
    use_cql=FLAGS.use_cql
    mixing_ratio = FLAGS.mixing_ratio

    total_grad_steps=0
    is_online=False
    online_eval_counter=-1
    do_eval=False
    online_rollout_timer = None
    train_timer = None
    epoch = 0
    train_metrics = None
    expl_metrics = None
    while True:
        metrics = {'epoch': epoch}

        if epoch == FLAGS.n_pretrain_epochs:
            is_online = True
            if FLAGS.cql_min_q_weight_online >= 0:
                print(f"changing cql alpha from {cql_min_q_weight} to {FLAGS.cql_min_q_weight_online}")
                cql_min_q_weight = FLAGS.cql_min_q_weight_online

            if not FLAGS.online_use_cql and use_cql:
                print("truning off cql during online phase and use sac")
                use_cql = False
                if sac.config.cql_lagrange:
                    model_keys = list(sac.model_keys)
                    model_keys.remove('log_alpha_prime')
                    sac._model_keys = tuple(model_keys)

        """
        Do evaluations when
        1. epoch = 0 to get initial performance
        2. every FLAGS.offline_eval_every_n_epoch for offline phase
        3. epoch == FLAGS.n_pretrain_epochs to get offline pre-trained performance
        4. every FLAGS.online_eval_every_n_env_steps for online phase
        5. when replay_buffer.total_steps >= FLAGS.max_online_env_steps to get final fine-tuned performance
        """
        do_eval = (epoch == 0 or (not is_online and epoch in offline_eval_epochs) or (epoch == FLAGS.n_pretrain_epochs) or (is_online and replay_buffer.total_steps // FLAGS.online_eval_every_n_env_steps > online_eval_counter) or (replay_buffer.total_steps >= FLAGS.max_online_env_steps))
            
        with Timer() as eval_timer:
            if do_eval:
                print(f"Starting Evaluation for Epoch {epoch}")
                trajs = eval_sampler.sample(
                    sampler_policy.update_params(sac.train_params['policy']),
                    FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['evaluation/average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['evaluation/average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                if use_goal:
                    # for adroit envs
                    metrics['evaluation/goal_achieved_rate'] = np.mean([1 in t['goal_achieved'] for t in trajs])


                if is_online:
                    online_eval_counter = replay_buffer.total_steps // FLAGS.online_eval_every_n_env_steps

                if FLAGS.save_model:
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, 'model.pkl')

                # logging the offline evaluation results
                if (epoch == 0) or (epoch in offline_eval_epochs) or (epoch == FLAGS.n_pretrain_epochs):
                    expr_now = datetime.now()
                    expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
                    offline_eval_file.write(f"{expr_time_now_str}\t{epoch}\t{metrics['evaluation/average_return']}\n")
                    offline_eval_file.flush()

                    if best_offline_agent_perf is None or best_offline_agent_perf < metrics['evaluation/average_return']:
                        best_offline_agent_epoch = epoch
                        best_offline_agent_perf = metrics['evaluation/average_return']
                        save_data = {'sac': sac, 'variant': variant, 'epoch': best_offline_agent_epoch, 'return': best_offline_agent_perf}
                        wandb_logger.save_pickle(save_data, best_offline_agent_filename)
                    
                    if epoch == FLAGS.n_pretrain_epochs:
                        offline_eval_file.write(f"**Best Policy**\t{best_offline_agent_epoch}\t{best_offline_agent_perf}\n")
                        offline_eval_file.flush()
                        offline_eval_file.close()
                        # since the best pooling is used in the offline validation, the best offline-optimized agent is used as the initial agent of the fine-tuning process
                        best_offline_agent = load_pickle(best_offline_agent_filepath)
                        if "sac" in best_offline_agent:
                            sac = best_offline_agent['sac']
                            sampler_policy = sampler_policy.update_params(sac.train_params['policy']) #SamplerPolicy(sac.policy, sac.train_params['policy'])
                
                if epoch >= FLAGS.n_pretrain_epochs:
                    # logging the fine-tuning evaluation results
                    expr_now = datetime.now()
                    expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
                    ft_eval_file.write(f"{expr_time_now_str}\t{epoch}\t{metrics['evaluation/average_return']}\n")
                    ft_eval_file.flush()

    
        metrics['grad_steps'] = total_grad_steps
        if is_online:
            metrics['env_steps'] = replay_buffer.total_steps
        metrics['epoch'] = epoch
        metrics['online_rollout_time'] = 0 if online_rollout_timer is None else online_rollout_timer()
        metrics['train_time'] = 0 if train_timer is None else train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = eval_timer() if train_timer is None else train_timer() + eval_timer()
        if FLAGS.n_pretrain_epochs >= 0:
            metrics['mixing_ratio'] = mixing_ratio
        if train_metrics is not None:
            metrics.update(train_metrics)
        if expl_metrics is not None:
            metrics.update(expl_metrics)
        
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=True)

        if replay_buffer.total_steps >= FLAGS.max_online_env_steps:
            print("Finished Training")
            ft_eval_file.flush()
            ft_eval_file.close()
            break

        with Timer() as online_rollout_timer:
            if is_online:
                print("collecting online trajs:", FLAGS.n_online_traj_per_epoch)
                trajs = train_sampler.sample(
                    sampler_policy.update_params(sac.train_params['policy']),
                    n_trajs=FLAGS.n_online_traj_per_epoch, deterministic=False, replay_buffer=replay_buffer
                )
                expl_metrics = {}
                expl_metrics['exploration/average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                expl_metrics['exploration/average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                if use_goal:
                    expl_metrics['exploration/goal_achieved_rate'] = np.mean([1 in t['goal_achieved'] for t in trajs])
        
        if train_timer is None:
            print("jit compiling train function: will take a while")
            
        with Timer() as train_timer:

            if FLAGS.n_pretrain_epochs >= 0 and epoch >= FLAGS.n_pretrain_epochs and FLAGS.online_utd_ratio > 0:
                n_train_step_per_epoch = np.sum([len(t["rewards"]) for t in trajs]) *  FLAGS.online_utd_ratio
            
            if FLAGS.n_pretrain_epochs >= 0:
                if FLAGS.mixing_ratio >= 0:
                    mixing_ratio = FLAGS.mixing_ratio
                else:
                    mixing_ratio = dataset['rewards'].shape[0] / (dataset['rewards'].shape[0] + replay_buffer.total_steps)
                batch_size_offline = int(FLAGS.batch_size * mixing_ratio)
                batch_size_online = FLAGS.batch_size - batch_size_offline

            for _ in range(n_train_step_per_epoch):
                if is_online:
                    # mix offline and online buffer
                    offline_batch = subsample_batch(dataset, batch_size_offline)
                    online_batch = replay_buffer.sample(batch_size_online)
                    batch = concatenate_batches([offline_batch, online_batch])
                    batch = batch_to_jax(batch)
                else:
                    # pure offline
                    batch = batch_to_jax(subsample_batch(dataset, FLAGS.batch_size))
                train_metrics = prefix_metrics(sac.train(batch, use_cql=use_cql, cql_min_q_weight=cql_min_q_weight, enable_calql=enable_calql), 'sac')
            total_grad_steps += n_train_step_per_epoch
        epoch += 1

if __name__ == '__main__':
    absl.app.run(main)
