import time
from typing import List, Tuple, Type, Union, Callable, Optional, Dict, Any

import gym
from gym import spaces
import torch as th
import torch.nn.functional as F

# Check if tensorboard is available for pytorch
# TODO: finish tensorboard integration
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     SummaryWriter = None
import numpy as np

from torchy_baselines.common import logger
from torchy_baselines.common.base_class import OffPolicyRLModel
from torchy_baselines.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn
from torchy_baselines.common.buffers import ReplayBuffer, ReplayBufferSamples
from torchy_baselines.common.utils import explained_variance, get_schedule_fn
from torchy_baselines.common.vec_env import VecEnv
from torchy_baselines.common.callbacks import BaseCallback
from torchy_baselines.common.noise import ActionNoise
from torchy_baselines.dqn.policies import DQNPolicy


class DQN(OffPolicyRLModel):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602
    Code: This implementation borrows code from probably Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)?

    :param policy: (PPOPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) The learning rate, it can be a function
        of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param batch_size: (int) Minibatch size for each gradient update
    :param gamma: (float) Discount factor
    :param epsilon_decay: (float) Decay factor for epsilon
    :param train_freq: (int) Update the model every ``train_freq`` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param epsilon: (float) Exploration factor for epsilon-greedy policy
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: (int) Seed for the pseudo random generators
    :param device: (str or th.device) Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(self, policy: Union[str, Type[DQNPolicy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Callable] = 3e-4,
                 buffer_size: int = 1000000,
                 learning_starts: int = 1000000,
                 batch_size: Optional[int] = 64,
                 gamma: float = 0.99,
                 epsilon_decay: float = 0.99,
                 train_freq: int = -1,
                 gradient_steps: int = 1,
                 epsilon: float = 0.05,
                 max_grad_norm: float = 10,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = 'auto',
                 _init_setup_model: bool = True):

        super(DQN, self).__init__(policy, env, DQNPolicy, learning_rate,
                                  buffer_size, learning_starts, batch_size,
                                  policy_kwargs, verbose, device, create_eval_env=create_eval_env, seed=seed)

        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.max_grad_norm = max_grad_norm
        self.tensorboard_log = tensorboard_log
        self.tb_writer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(DQN, self)._setup_model()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:

        # Update learning rate according to lr schedule
        self._update_learning_rate(self.policy.optimizer)

        # Decay learning rate
        self.policy.epsilon *= self.epsilon_decay

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the target Q values
                target_q = self.policy(replay_data.next_observations)
                target_q, _ = target_q.max(1)
                target_q = target_q.reshape(-1, 1)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates
            current_q = self.policy(replay_data.observations)

            # Gather q_values of our actions
            indices = replay_data.actions.long()
            current_q = th.gather(current_q, 1, indices)

            # Compute q loss
            loss = F.mse_loss(current_q, target_q)

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        logger.logkv("n_updates", self._n_updates)

    def learn(self,
              total_timesteps: int,
              callback: MaybeCallback = None,
              log_interval: int = 4,
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "DQN",
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> OffPolicyRLModel:

        episode_num, obs, callback = self._setup_learn(eval_env, callback, eval_freq,
                                                       n_eval_episodes, eval_log_path, reset_num_timesteps)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            rollout = self.collect_rollouts(self.env,
                                            n_steps=self.train_freq, n_episodes=-1, action_noise=self.action_noise,
                                            callback=callback,
                                            learning_starts=self.learning_starts,
                                            replay_buffer=self.replay_buffer,
                                            obs=obs, episode_num=episode_num,
                                            log_interval=log_interval)

            if rollout.continue_training is False:
                break

            obs = rollout.obs
            episode_num += rollout.n_episodes
            self._update_current_progress(self.num_timesteps, total_timesteps)

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                self.train(batch_size=self.batch_size,gradient_steps=self.gradient_steps)

        callback.on_training_end()

        return self

    def collect_rollouts(self,
                         env: VecEnv,
                         # Type hint as string to avoid circular import
                         callback: 'BaseCallback',
                         n_episodes: int = 1,
                         n_steps: int = -1,
                         action_noise: Optional[ActionNoise] = None,
                         learning_starts: int = 0,
                         epsilon: float = 1e-3,
                         replay_buffer: Optional[ReplayBuffer] = None,
                         obs: Optional[np.ndarray] = None,
                         episode_num: int = 0,
                         log_interval: Optional[int] = None) -> RolloutReturn:
        """
        Collect rollout using the current policy (and possibly fill the replay buffer)

        :param env: (VecEnv) The training environment
        :param n_episodes: (int) Number of episodes to use to collect rollout data
            You can also specify a `n_steps` instead
        :param n_steps: (int) Number of steps to use to collect rollout data
            You can also specify a `n_episodes` instead.
        :param action_noise: (Optional[ActionNoise]) Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param callback: (BaseCallback) Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param learning_starts: (int) Number of steps before learning for the warm-up phase.
        :param epsilon: (float) Epsilon to be used for epsilon greedy policy in case of discrete action space
        :param replay_buffer: (ReplayBuffer)
        :param obs: (np.ndarray) Last observation from the environment
        :param episode_num: (int) Episode index
        :param log_interval: (int) Log data every `log_interval` episodes
        :return: (RolloutReturn)
        """
        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyRLModel only support single environment"

        # Retrieve unnormalized observation for saving into the buffer
        if self._vec_normalize_env is not None:
            obs_ = self._vec_normalize_env.get_original_obs()

        self.rollout_data = None

        callback.on_rollout_start()
        continue_training = True

        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:
                # epsilon greedy exporation here
                if self.num_timesteps < learning_starts:
                    policy_action = np.array([self.action_space.sample()])
                else:
                    policy_action, _ = self.predict(obs, deterministic=False)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(policy_action)

                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, total_steps, total_episodes, None, continue_training=False)

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
                        obs_, new_obs_, reward_ = obs, new_obs, reward

                    replay_buffer.add(obs_, new_obs_, policy_action , reward_, done)

                obs = new_obs
                # Save the true unnormalized observation
                # otherwise obs_ = self._vec_normalize_env.unnormalize_obs(obs)
                # is a good approximation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                self.num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1
                if 0 < n_steps <= total_steps:
                    break

            if done:
                total_episodes += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                # Display training infos
                if self.verbose >= 1 and log_interval is not None and (
                        episode_num + total_episodes) % log_interval == 0:
                    fps = int(self.num_timesteps / (time.time() - self.start_time))
                    logger.logkv("episodes", episode_num + total_episodes)
                    if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                        logger.logkv('ep_rew_mean', self.safe_mean([ep_info['r'] for ep_info in self.ep_info_buffer]))
                        logger.logkv('ep_len_mean', self.safe_mean([ep_info['l'] for ep_info in self.ep_info_buffer]))
                    # logger.logkv("n_updates", n_updates)
                    logger.logkv("fps", fps)
                    logger.logkv("epsilon", self.policy.epsilon)
                    logger.logkv('time_elapsed', int(time.time() - self.start_time))
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()

        callback.on_rollout_end()

        return RolloutReturn(0.0, total_steps, total_episodes, obs, continue_training)


    def get_torch_variables(self) -> Tuple[List[str], List[str]]:
        """
        cf base class
        """
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
