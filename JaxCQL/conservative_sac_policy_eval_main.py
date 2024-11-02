import os
import numpy as np
import gym
import d4rl
from datetime import datetime
from mlxu.logging import load_pickle

import absl.app
import absl.flags

from .conservative_sac import ConservativeSAC
from .replay_buffer import subsample_batch, concatenate_batches, get_d4rl_dataset_with_mc_calculation, get_hand_dataset_with_mc_calculation
from .jax_utils import batch_to_jax
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import TrajSampler
from .utils import (
    Timer, define_flags_with_default, set_random_seed,
    get_user_flags, prefix_metrics, WandBLogger
)
from .jax_utils import next_rng
from viskit.logging import logger, setup_logger
from .replay_buffer import ReplayBuffer



# Calculate the distance (expected normalized error - mse) between the critic's estimate and the MC estimate for each (s, a)
def calculate_distance(sac, trajs, rng=None):
    errors = []
    
    # Iterate over each trajectory
    for traj in trajs:
        states = traj['observations']
        actions = traj['actions']
        mc_returns = traj['mc_returns']  # Use precomputed Monte Carlo returns

        # Get critic's Q-value estimates for all (state, action) pairs in the trajectory using sac.q_values()
        q_values_critic = np.asarray(sac.q_values(states, actions, rng))  # Use sac.q_values() to get Q-values
        # Compare the critic's estimate with the Monte Carlo return for each state-action pair
        for t, mc_return in enumerate(mc_returns):
            q_value_critic = q_values_critic[t]
            # Calculate the normalized error by dividing the absolute error by the MC return
            normalized_error = np.abs(q_value_critic - mc_return) / np.abs(mc_return)
            errors.append(normalized_error)
    
    return np.mean(errors)  # Return the average normalized error (distance)




FLAGS_DEF = define_flags_with_default(
    env='antmaze-medium-diverse-v2',
    seed=42,
    eval_dataset_fp="",
    pretrained_agent_fp="",
    alignment_episodes=100,
    max_epochs = 10,
    critic_utd_ratio = 1, # 0: size of training dataset, 1: size of training dataset / batch size
    save_model=False,
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


def main(argv):
    FLAGS = absl.flags.FLAGS

    now = datetime.now()
    expr_time_str = now.strftime("%Y%m%d-%H%M%S")
    algo = "SACCritic"
    criticUTD = 1 if FLAGS.critic_utd_ratio else "B"
    experiment_result_folder_name = f"ft_{algo}_{FLAGS.env}_seed{FLAGS.seed}_{FLAGS.alignment_episodes}episodes_{FLAGS.max_epochs}epochs_criticUTD{criticUTD}_{expr_time_str}"
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
    
    if FLAGS.env in ["pen-binary-v0", "door-binary-v0", "relocate-binary-v0"]:
        import mj_envs
        dataset = get_hand_dataset_with_mc_calculation(FLAGS.env, gamma=FLAGS.cql.discount, reward_scale=FLAGS.reward_scale, reward_bias=FLAGS.reward_bias, clip_action=FLAGS.clip_action)
        use_goal = True
    else:
        dataset = get_d4rl_dataset_with_mc_calculation(FLAGS.env, FLAGS.reward_scale, FLAGS.reward_bias, FLAGS.clip_action, gamma=FLAGS.cql.discount)
        use_goal = False

    assert dataset["next_observations"].shape == dataset["observations"].shape

    set_random_seed(FLAGS.seed)
    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, use_goal, gamma=FLAGS.cql.discount)
    train_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, use_goal, use_mc=True, gamma=FLAGS.cql.discount, reward_scale=FLAGS.reward_scale, reward_bias=FLAGS.reward_bias,)


    # load the evaluation dataset and the pretrained sac model
    eval_buffer = load_pickle(FLAGS.eval_dataset_fp)
    #eval_trajs = eval_buffer.get_trajs()
    eval_dataset_size = len(eval_buffer)
    eval_dataset = {
        'observations': eval_buffer._observations[:eval_dataset_size],
        'actions': eval_buffer._actions[:eval_dataset_size],
        'rewards': eval_buffer._rewards[:eval_dataset_size],
        'next_observations': eval_buffer._next_observations[:eval_dataset_size],
        'dones': eval_buffer._dones[:eval_dataset_size],
        'mc_returns': eval_buffer._mc_returns[:eval_dataset_size]
    }
    pretrained_sac = load_pickle(FLAGS.pretrained_agent_fp)["sac"] 
    #training_episodes_list = [100, 200, 500, 1000, 2000, 5000, 10000]

    alignment_env = gym.make(FLAGS.env)
    replay_buffer = ReplayBuffer(FLAGS.alignment_episodes*alignment_env.spec.max_episode_steps)
    train_dataset = None


    observation_dim = eval_sampler.env.observation_space.shape[0]
    action_dim = eval_sampler.env.action_space.shape[0]

    policy = TanhGaussianPolicy(
        observation_dim, action_dim, FLAGS.policy_arch, FLAGS.orthogonal_init,
        FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset
    )

    qf = FullyConnectedQFunction(observation_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init)

    sac = ConservativeSAC(FLAGS.cql, policy, qf)
    sac.policy = pretrained_sac.policy
    sac._train_states['policy'] = pretrained_sac._train_states['policy']
    # use the pretrained policy to collect new expisodes for training
    sampler_policy = SamplerPolicy(sac.policy, sac.train_params['policy'])

    viskit_metrics = {}

    total_grad_steps=0
    online_rollout_timer = None
    train_timer = None
    epoch = 0
    train_metrics = None
    expl_metrics = None
    alignment_dataset_size = None

    best_perf_be = None
    best_perf_re = None

    while True:
        metrics = {'epoch': epoch}

        # before learning the critic, check its Bellman error and averaged normalized mse on the evaluation dataset
        #distance = calculate_distance(sac, eval_trajs, next_rng())
        with Timer() as eval_timer:
            bellman_error, rae_error = sac.eval_critic(next_rng(), eval_dataset)        
            print(f"Evaluation Error: {rae_error} (rae error), {bellman_error} (bellman error)")
            metrics['evaluation/Q_Pi_Distance'] = rae_error
            metrics['evaluation/bellman_error'] = bellman_error
            # save the best model in terms of the smallest bellman error
            if FLAGS.save_model:
                if (best_perf_be is None) or (best_perf_be > bellman_error):
                    best_perf_be = bellman_error
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    save_data_fp = os.path.join(FLAGS.logging.output_dir, 'best_be_model.pkl')
                    wandb_logger.save_pickle(save_data, save_data_fp)

                if (best_perf_re is None) or (best_perf_re > rae_error):
                    best_perf_re = rae_error
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    save_data_fp = os.path.join(FLAGS.logging.output_dir, 'best_re_model.pkl')
                    wandb_logger.save_pickle(save_data, save_data_fp)

        # collect episodes for training the true Q function for the policy
        with Timer() as online_rollout_timer:
            if epoch == 0: # collect new episodes as the training dataset before starting the training
                print("collecting online trajs:", FLAGS.alignment_episodes)
                trajs = train_sampler.sample(
                    sampler_policy,
                    n_trajs=FLAGS.alignment_episodes, deterministic=False, replay_buffer=replay_buffer
                )
                expl_metrics = {}
                expl_metrics['exploration/average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                expl_metrics['exploration/average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                if use_goal:
                    expl_metrics['exploration/goal_achieved_rate'] = np.mean([1 in t['goal_achieved'] for t in trajs])
                alignment_dataset_size = np.sum([len(t["rewards"]) for t in trajs])
                if FLAGS.critic_utd_ratio == 1:
                     n_train_step_per_epoch = alignment_dataset_size//FLAGS.batch_size if alignment_dataset_size%FLAGS.batch_size==0 else (alignment_dataset_size//FLAGS.batch_size + 1)
                else:
                    n_train_step_per_epoch = alignment_dataset_size

            train_dataset_size = len(replay_buffer)
            train_dataset = {
                'observations': replay_buffer._observations[:train_dataset_size],
                'actions': replay_buffer._actions[:train_dataset_size],
                'rewards': replay_buffer._rewards[:train_dataset_size],
                'next_observations': replay_buffer._next_observations[:train_dataset_size],
                'dones': replay_buffer._dones[:train_dataset_size],
                'mc_returns': replay_buffer._mc_returns[:train_dataset_size]
            }

         
        metrics['grad_steps'] = total_grad_steps
        metrics['env_steps'] = replay_buffer.total_steps
        metrics['epoch'] = epoch
        metrics['online_rollout_time'] = 0 if epoch!=0 else online_rollout_timer()
        metrics['train_time'] = 0 if train_timer is None else train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = eval_timer() if train_timer is None else train_timer() + eval_timer()
        if train_metrics is not None:
            metrics.update(train_metrics)
        if expl_metrics is not None:
            metrics.update(expl_metrics)
        
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

        if epoch >= FLAGS.max_epochs:
            print("Finished Training")
            break

       
        if train_timer is None:
            print("jit compiling train function: will take a while")
            
        with Timer() as train_timer:
            train_metrics_prefix = 'sac' 
            for _ in range(n_train_step_per_epoch):
                batch = replay_buffer.sample(FLAGS.batch_size)
                batch = batch_to_jax(batch)
                train_metrics = prefix_metrics(sac.train_critic(batch), train_metrics_prefix)
            total_grad_steps += n_train_step_per_epoch

            # evaluate the critic on the training dataset
            bellman_error, rae_error = sac.eval_critic(next_rng(), train_dataset)        
            print(f"Training Error: {rae_error} (rae error), {bellman_error} (bellman error)")
            train_metrics[f'{train_metrics_prefix}/Q_Pi_Distance'] = rae_error
            train_metrics[f'{train_metrics_prefix}/bellman_error'] = bellman_error

        epoch += 1

    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        save_data_fp = os.path.join(FLAGS.logging.output_dir, 'final_model.pkl')
        wandb_logger.save_pickle(save_data, save_data_fp)



if __name__ == '__main__':
    absl.app.run(main)
