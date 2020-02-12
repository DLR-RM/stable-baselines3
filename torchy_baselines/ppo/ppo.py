import os
import time
from typing import Optional, Tuple, List

import gym
from gym import spaces
import torch as th
import torch.nn.functional as F

# Check if tensorboard is available for pytorch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
import numpy as np

from torchy_baselines.common.base_class import BaseRLModel
from torchy_baselines.common.buffers import RolloutBuffer
from torchy_baselines.common.utils import explained_variance, get_schedule_fn
from torchy_baselines.common.vec_env import VecEnv
from torchy_baselines.common.callbacks import BaseCallback
from torchy_baselines.common import logger
from torchy_baselines.ppo.policies import PPOPolicy


class PPO(BaseRLModel):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: (PPOPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) The learning rate, it can be a function
        of the current progress (from 1 to 0)
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: (int) Minibatch size
    :param n_epochs: (int) Number of epoch when optimizing the surrogate loss
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: (float or callable) Clipping parameter, it can be a function of the current progress
        (from 1 to 0).
    :param clip_range_vf: (float or callable) Clipping parameter for the value function,
        it can be a function of the current progress (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param use_sde: (bool) Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: (int) Sample a new noise matrix every n steps when using SDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: (float) Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param seed: (int) Seed for the pseudo random generators
    :param device: (str or th.device) Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(self, policy, env, learning_rate=3e-4,
                 n_steps=2048, batch_size=64, n_epochs=10,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 use_sde=False, sde_sample_freq=-1,
                 target_kl=None, tensorboard_log=None, create_eval_env=False,
                 policy_kwargs=None, verbose=0, seed=0, device='auto',
                 _init_setup_model=True):

        super(PPO, self).__init__(policy, env, PPOPolicy, policy_kwargs=policy_kwargs,
                                  verbose=verbose, device=device, use_sde=use_sde, sde_sample_freq=sde_sample_freq,
                                  create_eval_env=create_eval_env, support_multi_env=True, seed=seed)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.target_kl = target_kl
        self.tensorboard_log = tensorboard_log
        self.tb_writer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        self._setup_learning_rate()
        # TODO: preprocessing: one hot vector for obs discrete
        state_dim = self.observation_space.shape[0]
        if isinstance(self.action_space, spaces.Box):
            # Action is a 1D vector
            action_dim = self.action_space.shape[0]
        elif isinstance(self.action_space, spaces.Discrete):
            # Action is a scalar
            action_dim = 1

        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(self.n_steps, state_dim, action_dim, self.device,
                                            gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=self.n_envs)
        self.policy = self.policy_class(self.observation_space, self.action_space,
                                        self.learning_rate, use_sde=self.use_sde, device=self.device,
                                        **self.policy_kwargs)
        self.policy = self.policy.to(self.device)

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def collect_rollouts(self,
                        env: VecEnv,
                        callback: BaseCallback,
                        rollout_buffer: RolloutBuffer,
                        n_rollout_steps: int = 256,
                        obs: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], bool]:

        n_steps = 0
        continue_training = True
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        # TODO: ensure episodic setting?
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:

            if callback() is False:
                continue_training = False
                return None, continue_training


            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                actions, values, log_probs = self.policy.forward(obs)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self._update_info_buffer(infos)
            n_steps += 1
            self.num_timesteps += env.num_envs

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(obs, actions, rewards, dones, values, log_probs)
            obs = new_obs

        rollout_buffer.compute_returns_and_advantage(values, dones=dones)

        callback.on_rollout_end()

        return obs, continue_training

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress)

        for gradient_step in range(gradient_steps):
            approx_kl_divs = []
            # Sample replay buffer
            for replay_data in self.rollout_buffer.get(batch_size):
                # Unpack
                obs, action, old_values, old_log_prob, advantage, return_batch = replay_data

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action for float to long
                    action = action.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(obs, action)
                values = values.flatten()
                # Normalize advantage
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - old_log_prob)
                # clipped surrogate loss
                policy_loss_1 = advantage * ratio
                policy_loss_2 = advantage * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = old_values + th.clamp(values - old_values, -clip_range_vf, clip_range_vf)
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(return_batch, values_pred)

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -log_prob.mean()
                else:
                    entropy_loss = -th.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                approx_kl_divs.append(th.mean(old_log_prob - log_prob).detach().cpu().numpy())

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print("Early stopping at step {} due to reaching max kl: {:.2f}".format(gradient_step,
                                                                                        np.mean(approx_kl_divs)))
                break

        explained_var = explained_variance(self.rollout_buffer.returns.flatten(),
                                           self.rollout_buffer.values.flatten())

        logger.logkv("clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.logkv("clip_range_vf", clip_range_vf)


        logger.logkv("explained_variance", explained_var)
        # TODO: gather stats for the entropy and other losses?
        logger.logkv("entropy_loss", entropy_loss.item())
        logger.logkv("policy_loss", policy_loss.item())
        logger.logkv("value_loss", value_loss.item())
        if hasattr(self.policy, 'log_std'):
            logger.logkv("std", th.exp(self.policy.log_std).mean().item())

    def learn(self, total_timesteps, callback=None, log_interval=1,
              eval_env=None, eval_freq=-1, n_eval_episodes=5, tb_log_name="PPO",
              eval_log_path=None, reset_num_timesteps=True):

        episode_num, obs, callback = self._setup_learn(eval_env, callback, eval_freq,
                                                       n_eval_episodes, eval_log_path, reset_num_timesteps)
        iteration = 0

        if self.tensorboard_log is not None and SummaryWriter is not None:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.tensorboard_log, tb_log_name))

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            obs, continue_training = self.collect_rollouts(self.env, callback,
                                                           self.rollout_buffer,
                                                           n_rollout_steps=self.n_steps,
                                                           obs=obs)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress(self.num_timesteps, total_timesteps)

            # Display training infos
            if self.verbose >= 1 and log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.logkv("iterations", iteration)
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.logkv('ep_rew_mean', self.safe_mean([ep_info['r'] for ep_info in self.ep_info_buffer]))
                    logger.logkv('ep_len_mean', self.safe_mean([ep_info['l'] for ep_info in self.ep_info_buffer]))
                logger.logkv("fps", fps)
                logger.logkv('time_elapsed', int(time.time() - self.start_time))
                logger.logkv("total timesteps", self.num_timesteps)
                logger.dumpkvs()

            self.train(self.n_epochs, batch_size=self.batch_size)

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
