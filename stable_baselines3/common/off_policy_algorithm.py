import time
import os
import pickle
import warnings
from typing import Union, Type, Optional, Dict, Any, Callable

import gym
import torch as th
import numpy as np

from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import GymEnv, RolloutReturn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer


class OffPolicyAlgorithm(BaseAlgorithm):
    """
    The base for Off-Policy algorithms (ex: SAC/TD3)

    :param policy: Policy object
    :param env: The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param policy_base: The base policy used by this method
    :param learning_rate: (float or callable) learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param batch_size: (int) Minibatch size for each gradient update
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: (bool) Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param sde_support: (bool) Whether the model support gSDE or not
    """

    def __init__(self,
                 policy: Type[BasePolicy],
                 env: Union[GymEnv, str],
                 policy_base: Type[BasePolicy],
                 learning_rate: Union[float, Callable],
                 buffer_size: int = int(1e6),
                 learning_starts: int = 100,
                 batch_size: int = 256,
                 policy_kwargs: Dict[str, Any] = None,
                 tensorboard_log: Optional[str] = None,
                 verbose: int = 0,
                 device: Union[th.device, str] = 'auto',
                 support_multi_env: bool = False,
                 create_eval_env: bool = False,
                 monitor_wrapper: bool = True,
                 seed: Optional[int] = None,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 use_sde_at_warmup: bool = False,
                 sde_support: bool = True):

        super(OffPolicyAlgorithm, self).__init__(policy=policy, env=env, policy_base=policy_base,
                                                 learning_rate=learning_rate, policy_kwargs=policy_kwargs,
                                                 tensorboard_log=tensorboard_log, verbose=verbose,
                                                 device=device, support_multi_env=support_multi_env,
                                                 create_eval_env=create_eval_env, monitor_wrapper=monitor_wrapper,
                                                 seed=seed, use_sde=use_sde, sde_sample_freq=sde_sample_freq)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.actor = None  # type: Optional[th.nn.Module]
        self.replay_buffer = None  # type: Optional[ReplayBuffer]
        # Update policy keyword arguments
        if sde_support:
            self.policy_kwargs['use_sde'] = self.use_sde
        self.policy_kwargs['device'] = self.device
        # For gSDE only
        self.use_sde_at_warmup = use_sde_at_warmup

    def _setup_model(self):
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.observation_space,
                                          self.action_space, self.device)
        self.policy = self.policy_class(self.observation_space, self.action_space,
                                        self.lr_schedule, **self.policy_kwargs)
        self.policy = self.policy.to(self.device)

    def save_replay_buffer(self, path: str):
        """
        Save the replay buffer as a pickle file.

        :param path: (str) Path to a log folder
        """
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        with open(os.path.join(path, 'replay_buffer.pkl'), 'wb') as file_handler:
            pickle.dump(self.replay_buffer, file_handler)

    def load_replay_buffer(self, path: str):
        """
        Load a replay buffer from a pickle file.

        :param path: (str) Path to the pickled replay buffer.
        """
        with open(path, 'rb') as file_handler:
            self.replay_buffer = pickle.load(file_handler)
        assert isinstance(self.replay_buffer, ReplayBuffer), 'The replay buffer must inherit from ReplayBuffer class'

    def collect_rollouts(self,  # noqa: C901
                         env: VecEnv,
                         # Type hint as string to avoid circular import
                         callback: 'BaseCallback',
                         n_episodes: int = 1,
                         n_steps: int = -1,
                         action_noise: Optional[ActionNoise] = None,
                         learning_starts: int = 0,
                         replay_buffer: Optional[ReplayBuffer] = None,
                         log_interval: Optional[int] = None) -> RolloutReturn:
        """
        Collect experiences and store them into a ReplayBuffer.

        :param env: (VecEnv) The training environment
        :param callback: (BaseCallback) Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param n_episodes: (int) Number of episodes to use to collect rollout data
            You can also specify a ``n_steps`` instead
        :param n_steps: (int) Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param action_noise: (Optional[ActionNoise]) Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: (int) Number of steps before learning for the warm-up phase.
        :param replay_buffer: (ReplayBuffer)
        :param log_interval: (int) Log data every ``log_interval`` episodes
        :return: (RolloutReturn)
        """
        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

        if n_episodes > 0 and n_steps > 0:
            # Note we are refering to the constructor arguments
            # that are named `train_freq` and `n_episodes_rollout`
            # but correspond to `n_steps` and `n_episodes` here
            warnings.warn("You passed a positive value for `train_freq` and `n_episodes_rollout`."
                          "Please make sure this is intended. "
                          "The agent will collect data by stepping in the environment "
                          "until both conditions are true: "
                          "`number of steps in the env` >= `train_freq` and "
                          "`number of episodes` > `n_episodes_rollout`")

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and total_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
                    # Warmup phase
                    unscaled_action = np.array([self.action_space.sample()])
                else:
                    # Note: we assume that the policy uses tanh to scale the action
                    # We use non-deterministic action in the case of SAC, for TD3, it does not matter
                    unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

                # Rescale the action from [low, high] to [-1, 1]
                if isinstance(self.action_space, gym.spaces.Box):
                    scaled_action = self.policy.scale_action(unscaled_action)

                    # Add noise to the action (improve exploration)
                    if action_noise is not None:
                        # NOTE: in the original implementation of TD3, the noise was applied to the unscaled action
                        # Update(October 2019): Not anymore
                        scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

                    # We store the scaled action in the buffer
                    buffer_action = scaled_action
                    action = self.policy.unscale_action(scaled_action)
                else:
                    # Discrete case, no need to normalize or clip
                    buffer_action = unscaled_action
                    action = buffer_action

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)

                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer
                if replay_buffer is not None:
                    # Store only the unnormalized version
                    if self._vec_normalize_env is not None:
                        new_obs_ = self._vec_normalize_env.get_original_obs()
                        reward_ = self._vec_normalize_env.get_original_reward()
                    else:
                        # Avoid changing the original ones
                        self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

                    replay_buffer.add(self._last_original_obs, new_obs_, buffer_action, reward_, done)

                self._last_obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    self._last_original_obs = new_obs_

                self.num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1
                if 0 < n_steps <= total_steps:
                    break

            if done:
                total_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    fps = int(self.num_timesteps / (time.time() - self.start_time))
                    logger.record("time/episodes", self._episode_num, exclude="tensorboard")
                    if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                        logger.record('rollout/ep_rew_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buffer]))
                        logger.record('rollout/ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buffer]))
                    logger.record("time/fps", fps)
                    logger.record('time/time_elapsed', int(time.time() - self.start_time), exclude="tensorboard")
                    logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
                    if self.use_sde:
                        logger.record("train/std", (self.actor.get_std()).mean().item())

                    if len(self.ep_success_buffer) > 0:
                        logger.record('rollout/success rate', safe_mean(self.ep_success_buffer))
                    # Pass the number of timesteps for tensorboard
                    logger.dump(step=self.num_timesteps)

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)
