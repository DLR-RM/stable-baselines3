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
from torchy_baselines.common.type_aliases import GymEnv, MaybeCallback
from torchy_baselines.common.buffers import ReplayBuffer, ReplayBufferSamples
from torchy_baselines.common.utils import explained_variance, get_schedule_fn
from torchy_baselines.common.vec_env import VecEnv
from torchy_baselines.common.callbacks import BaseCallback
from torchy_baselines.ppo.policies import PPOPolicy


class DQN(OffPolicyRLModel):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/pdf/1312.5602.pdf
    Code: This implementation borrows code from

    :param policy: (PPOPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) The learning rate, it can be a function
        of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param batch_size: (int) Minibatch size for each gradient update
    :param gamma: (float) Discount factor
    :param epsilon: (float) Exploration factor for epsilon-greedy policy
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

    def __init__(self, policy: Union[str, Type[PPOPolicy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Callable] = 3e-4,
                 buffer_size: int = 1000000,
                 learning_starts: int = 1000000,
                 batch_size: Optional[int] = 64,
                 gamma: float = 0.99,
                 epsilon: float = 0.05,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = 'auto',
                 _init_setup_model: bool = True):

        super(DQN, self).__init__(policy, env, policy, learning_rate,
                                  buffer_size, learning_starts, batch_size,
                                  policy_kwargs, verbose, device, create_eval_env=create_eval_env, seed=seed)

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_buffer = ReplayBuffer()
        self.tensorboard_log = tensorboard_log
        self.tb_writer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(DQN, self)._setup_model()

    def init_replay_buffer(self,
                           env: gym.Env,
                           replay_buffer: ReplayBuffer):
        """
        Initially fills the replay buffer with random samples from the environment
        (only non vector environments supported)

        :param env: (Env) Environments to sample from
        :param replay_buffer: (ReplayBuffer) ReplayBuffer to fill
        """
        replay_buffer.reset()
        obs = env.reset()

        while not replay_buffer.full:

            action = env.action_space.sample()
            # Perform action
            new_obs, reward, done, info = env.step(action)

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                action = action.reshape(-1, 1)
            replay_buffer.add(obs, new_obs, action, reward, done)
            if done:
                obs = env.reset()
            else:
                obs = new_obs.copy()  # copy the observation here to avoid conflicts

    def train(self, gradient_steps: int, batch_size: int = 100, policy_delay: int = 2) -> None:

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        for gradient_step in range(gradient_steps):

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the target Q value
                target_q1, target_q2 = self.critic_target(replay_data.next_observations, next_actions)
                target_q = th.min(target_q1, target_q2)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates
            current_q1, current_q2 = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if gradient_step % policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations,
                                                     self.actor(replay_data.observations)).mean()

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                # Update the frozen target networks
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self._n_updates += gradient_steps
        logger.logkv("n_updates", self._n_updates)

    def train(self, n_epochs: int, batch_size: int = 64) -> None:
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # train for gradient_steps epochs
        for epoch in range(n_epochs):

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # forward observations to get q values
            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(next_obs).to(self.device)
                actions = self.policy.forward(obs_tensor)
            action = action.cpu().numpy()
            # some how get best q value from that...
            # compute targets
            target = rewards + self.gamma * dones * best_q
            self.policy.optimizer.zero_grad()
            output = self.policy(obs)
            loss = self.policy.criterion(output, target)
            loss.backward()
            self.policy.optimizer.step()

        self._n_updates += 1
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
                                            n_steps=self.train_freq, action_noise=self.action_noise,
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
                self.train(batch_size=self.batch_size)

        callback.on_training_end()

        return self

    def get_torch_variables(self) -> Tuple[List[str], List[str]]:
        """
        cf base class
        """
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
