import time

import numpy as np
import torch as th
import torch.nn.functional as F
from typing import List, Tuple, Type, Union, Callable, Optional, Dict, Any

from stable_baselines3.common import logger
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.td3.policies import TD3Policy


class TD3(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: (TD3Policy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (GymEnv or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: (float) the discount factor
    :param train_freq: (int) Update the model every ``train_freq`` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param n_episodes_rollout: (int) Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``
    :param action_noise: (ActionNoise) the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param policy_delay: (int) Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: (float) Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: (float) Limit for absolute value of target policy smoothing noise.
    :param use_sde: (bool) Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: (int) Sample a new noise matrix every n steps when using SDE
        Default: -1 (only sample at the beginning of the rollout)
    :param sde_max_grad_norm: (float)
    :param sde_ent_coef: (float)
    :param sde_log_std_scheduler: (callable)
    :param use_sde_at_warmup: (bool) Whether to use SDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: (int) Seed for the pseudo random generators
    :param device: (str or th.device) Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(self, policy: Union[str, Type[TD3Policy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Callable] = 1e-3,
                 buffer_size: int = int(1e6),
                 learning_starts: int = 100,
                 batch_size: int = 100,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 train_freq: int = -1,
                 gradient_steps: int = -1,
                 n_episodes_rollout: int = 1,
                 action_noise: Optional[ActionNoise] = None,
                 policy_delay: int = 2,
                 target_policy_noise: float = 0.2,
                 target_noise_clip: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 sde_max_grad_norm: float = 1,
                 sde_ent_coef: float = 0.0,
                 sde_log_std_scheduler: Optional[Callable] = None,
                 use_sde_at_warmup: bool = False,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Dict[str, Any] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = 'auto',
                 _init_setup_model: bool = True):

        super(TD3, self).__init__(policy, env, TD3Policy, learning_rate,
                                  buffer_size, learning_starts, batch_size,
                                  policy_kwargs, tensorboard_log, verbose, device,
                                  create_eval_env=create_eval_env, seed=seed,
                                  use_sde=use_sde, sde_sample_freq=sde_sample_freq,
                                  use_sde_at_warmup=use_sde_at_warmup)

        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.n_episodes_rollout = n_episodes_rollout
        self.tau = tau
        self.gamma = gamma
        self.action_noise = action_noise
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        # State Dependent Exploration
        self.sde_max_grad_norm = sde_max_grad_norm
        self.sde_ent_coef = sde_ent_coef
        self.sde_log_std_scheduler = sde_log_std_scheduler
        self.on_policy_exploration = True
        self.sde_vf = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(TD3, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.vf_net = self.policy.vf_net

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
        logger.record("train/n_updates", self._n_updates, exclude='tensorboard')

    def train_sde(self) -> None:
        # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)

        # Unpack
        obs, action, advantage, returns = [self.rollout_data[key] for key in
                                           ['observations', 'actions', 'advantage', 'returns']]

        log_prob, entropy = self.actor.evaluate_actions(obs, action)
        values = self.vf_net(obs).flatten()

        # Normalize advantage
        # if self.normalize_advantage:
        #     advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(returns, values)

        # A2C loss
        policy_loss = -(advantage * log_prob).mean()  # pytype: disable=attribute-error

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -log_prob.mean()
        else:
            entropy_loss = -th.mean(entropy)

        vf_coef = 0.5
        loss = policy_loss + self.sde_ent_coef * entropy_loss + vf_coef * value_loss

        # Optimization step
        self.actor.sde_optimizer.zero_grad()
        loss.backward()

        assert not th.isnan(log_prob).any(), log_prob
        assert not th.isnan(entropy).any()
        assert not th.isnan(self.actor.log_std.grad).any()
        assert not th.isnan(self.actor.log_std).any()

        # Clip grad norm
        th.nn.utils.clip_grad_norm_([self.actor.log_std], self.sde_max_grad_norm)
        self.actor.sde_optimizer.step()

        del self.rollout_data

    def learn(self,
              total_timesteps: int,
              callback: MaybeCallback = None,
              log_interval: int = 4,
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "TD3",
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> OffPolicyAlgorithm:

        total_timesteps, callback = self._setup_learn(total_timesteps, eval_env, callback, eval_freq,
                                                      n_eval_episodes, eval_log_path, reset_num_timesteps,
                                                      tb_log_name)
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            rollout = self.collect_rollouts(self.env, n_episodes=self.n_episodes_rollout,
                                            n_steps=self.train_freq, action_noise=self.action_noise,
                                            callback=callback,
                                            learning_starts=self.learning_starts,
                                            replay_buffer=self.replay_buffer,
                                            log_interval=log_interval)

            if rollout.continue_training is False:
                break

            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:

                if self.use_sde:
                    if self.sde_log_std_scheduler is not None:
                        # Call the scheduler
                        value = self.sde_log_std_scheduler(self._current_progress_remaining)
                        self.actor.log_std.data = th.ones_like(self.actor.log_std) * value
                    else:
                        # On-policy gradient
                        self.train_sde()

                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                self.train(gradient_steps, batch_size=self.batch_size, policy_delay=self.policy_delay)

        callback.on_training_end()

        return self

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
        Collect rollout using the current policy (and possibly fill the replay buffer)

        :param env: (VecEnv) The training environment
        :param n_episodes: (int) Number of episodes to use to collect rollout data
            You can also specify a ``n_steps`` instead
        :param n_steps: (int) Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param action_noise: (Optional[ActionNoise]) Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param callback: (BaseCallback) Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param learning_starts: (int) Number of steps before learning for the warm-up phase.
        :param replay_buffer: (ReplayBuffer)
        :param log_interval: (int) Log data every ``log_interval`` episodes
        :return: (RolloutReturn)
        """
        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyRLModel only support single environment"

        self.rollout_data = None
        if self.use_sde:
            self.actor.reset_noise()
            # Reset rollout data
            if self.on_policy_exploration:
                self.rollout_data = {key: [] for key in ['observations', 'actions', 'rewards', 'dones', 'values']}

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
                scaled_action = self.policy.scale_action(unscaled_action)

                if self.use_sde:
                    # When using SDE, the action can be out of bounds
                    # TODO: fix with squashing and account for that in the proba distribution
                    clipped_action = np.clip(scaled_action, -1, 1)
                else:
                    clipped_action = scaled_action

                # Add noise to the action (improve exploration)
                if action_noise is not None:
                    # NOTE: in the original implementation of TD3, the noise was applied to the unscaled action
                    # Update(October 2019): Not anymore
                    clipped_action = np.clip(clipped_action + action_noise(), -1, 1)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(self.policy.unscale_action(clipped_action))

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

                    replay_buffer.add(self._last_original_obs, new_obs_, clipped_action, reward_, done)

                if self.rollout_data is not None:
                    # Assume only one env
                    self.rollout_data['observations'].append(self._last_obs[0].copy())
                    self.rollout_data['actions'].append(scaled_action[0].copy())
                    self.rollout_data['rewards'].append(reward[0].copy())
                    self.rollout_data['dones'].append(done[0].copy())
                    obs_tensor = th.FloatTensor(self._last_obs).to(self.device)
                    self.rollout_data['values'].append(self.vf_net(obs_tensor)[0].cpu().detach().numpy())

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

        # Post processing
        if self.rollout_data is not None:
            for key in ['observations', 'actions', 'rewards', 'dones', 'values']:
                self.rollout_data[key] = th.FloatTensor(np.array(self.rollout_data[key])).to(self.device)

            self.rollout_data['returns'] = self.rollout_data['rewards'].clone()  # pytype: disable=attribute-error
            self.rollout_data['advantage'] = self.rollout_data['rewards'].clone()  # pytype: disable=attribute-error

            # Compute return and advantage
            last_return = 0.0
            for step in reversed(range(len(self.rollout_data['rewards']))):
                if step == len(self.rollout_data['rewards']) - 1:
                    next_non_terminal = 1.0 - done[0]
                    next_value = self.vf_net(th.FloatTensor(self._last_obs).to(self.device))[0].detach()
                    last_return = self.rollout_data['rewards'][step] + next_non_terminal * next_value
                else:
                    next_non_terminal = 1.0 - self.rollout_data['dones'][step + 1]
                    last_return = self.rollout_data['rewards'][step] + self.gamma * last_return * next_non_terminal
                self.rollout_data['returns'][step] = last_return
            self.rollout_data['advantage'] = self.rollout_data['returns'] - self.rollout_data['values']

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)

    def excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded by default
        when saving the model.

        :return: (List[str]) List of parameters that should be excluded from save
        """
        # Exclude aliases
        return super(TD3, self).excluded_save_params() + ["actor", "critic", "vf_net", "actor_target", "critic_target"]

    def get_torch_variables(self) -> Tuple[List[str], List[str]]:
        """
        cf base class
        """
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []
