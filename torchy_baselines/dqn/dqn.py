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
from torchy_baselines.common.base_class import BaseRLModel
from torchy_baselines.common.type_aliases import GymEnv, MaybeCallback
from torchy_baselines.common.buffers import ReplayBuffer, ReplayBufferSamples
from torchy_baselines.common.utils import explained_variance, get_schedule_fn
from torchy_baselines.common.vec_env import VecEnv
from torchy_baselines.common.callbacks import BaseCallback
from torchy_baselines.ppo.policies import PPOPolicy


class DQN(BaseRLModel):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/pdf/1312.5602.pdf
    Code: This implementation borrows code from

    :param policy: (PPOPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) The learning rate, it can be a function
        of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: (int) Minibatch size
    :param n_epochs: (int) Number of epoch when optimizing the surrogate loss
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
                 n_steps: int = 2048,
                 batch_size: Optional[int] = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 epsilon: float = 0.05,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = 'auto',
                 _init_setup_model: bool = True):

        super(DQN, self).__init__(policy, env, PPOPolicy, learning_rate, policy_kwargs=policy_kwargs,
                                  verbose=verbose, device=device, create_eval_env=create_eval_env,
                                  support_multi_env=True, seed=seed)

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer()
        self.tensorboard_log = tensorboard_log
        self.tb_writer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.observation_space, self.action_space, self.device,
                                          n_envs=self.n_envs)
        self.policy = self.policy_class(self.observation_space, self.action_space,
                                        self.lr_schedule, device=self.device,
                                        **self.policy_kwargs)
        self.policy = self.policy.to(self.device)

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

    def train(self, n_epochs: int, batch_size: int = 64) -> None:
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # train for gradient_steps epochs
        for epoch in range(n_epochs):

            # sample random minibatch from replay buffer
            obs, actions, next_obs, dones, rewards = self.replay_buffer.sample(batch_size)
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

        self._n_updates += n_epochs
        explained_var = explained_variance(self.rollout_buffer.returns.flatten(),
                                           self.rollout_buffer.values.flatten())

        logger.logkv("n_updates", self._n_updates)
        logger.logkv("clip_fraction", np.mean(clip_fraction))
        logger.logkv("clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.logkv("clip_range_vf", clip_range_vf)

        logger.logkv("approx_kl", np.mean(approx_kl_divs))
        logger.logkv("explained_variance", explained_var)
        logger.logkv("entropy_loss", np.mean(entropy_losses))
        logger.logkv("policy_gradient_loss", np.mean(pg_losses))
        logger.logkv("value_loss", np.mean(value_losses))
        if hasattr(self.policy, 'log_std'):
            logger.logkv("std", th.exp(self.policy.log_std).mean().item())

    def learn(self,
              total_timesteps: int,
              callback: MaybeCallback = None,
              log_interval: int = 1,
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "DQN",
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> 'DQN':

        episode_num, obs, callback = self._setup_learn(eval_env, callback, eval_freq,
                                                       n_eval_episodes, eval_log_path, reset_num_timesteps)
        iteration = 0

        # if self.tensorboard_log is not None and SummaryWriter is not None:
        #     self.tb_writer = SummaryWriter(log_dir=os.path.join(self.tensorboard_log, tb_log_name))

        callback.on_training_start(locals(), globals())

        # prepare replay_buffer
        self.init_replay_buffer(self.env, self.replay_buffer)

        logger.info("Replay buffer filled")

        obs = self.env.reset()

        while self.num_timesteps < total_timesteps:

            # step environment epsilon greedy and update replay buffer
            if np.random.random_sample() < self.epsilon:
                action = self.action_space.sample()
            else:
                with th.no_grad():
                    # Convert to pytorch tensor
                    obs_tensor = th.as_tensor(obs).to(self.device)
                    action = self.policy.forward(obs_tensor)
                action = action.cpu().numpy()
            new_obs, reward, done, info = self.env.step(action)
            self.replay_buffer.add(obs, new_obs, action, reward, done)

            # perform gradient step on new batch
            self.train(self.n_epochs, batch_size=self.batch_size)

            iteration += 1
            self._update_current_progress(self.num_timesteps, total_timesteps)
            # Display training infos
            if self.verbose >= 1 and log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.logkv("iterations", iteration)
                logger.logkv("fps", fps)
                logger.logkv('time_elapsed', int(time.time() - self.start_time))
                logger.logkv("total timesteps", self.num_timesteps)
                logger.dumpkvs()
            # For tensorboard integration
            # if self.tb_writer is not None:
            #     self.tb_writer.add_scalar('Eval/reward', mean_reward, self.num_timesteps)

        callback.on_training_end()

        return self

    def get_torch_variables(self) -> Tuple[List[str], List[str]]:
        """
        cf base class
        """
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
