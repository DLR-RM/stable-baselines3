import torch as th
import torch.nn.functional as F
from typing import List, Tuple, Type, Union, Callable, Optional, Dict, Any

from torchy_baselines.common import logger
from torchy_baselines.common.base_class import OffPolicyRLModel
from torchy_baselines.common.buffers import ReplayBuffer
from torchy_baselines.common.noise import ActionNoise
from torchy_baselines.common.type_aliases import ReplayBufferSamples, GymEnv, MaybeCallback
from torchy_baselines.td3.policies import TD3Policy


class TD3(OffPolicyRLModel):
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
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)
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

        super(TD3, self).__init__(policy, env, TD3Policy, policy_kwargs, verbose, device,
                                  create_eval_env=create_eval_env, seed=seed,
                                  use_sde=use_sde, sde_sample_freq=sde_sample_freq,
                                  use_sde_at_warmup=use_sde_at_warmup)

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.n_episodes_rollout = n_episodes_rollout
        self.batch_size = batch_size
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
        self._setup_learning_rate()
        obs_dim, action_dim = self.observation_space.shape[0], self.action_space.shape[0]
        self.set_random_seed(self.seed)
        self.replay_buffer = ReplayBuffer(self.buffer_size, obs_dim, action_dim, self.device)
        self.policy = self.policy_class(self.observation_space, self.action_space,
                                        self.lr_schedule, use_sde=self.use_sde,
                                        device=self.device, **self.policy_kwargs)
        self.policy = self.policy.to(self.device)
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.vf_net = self.policy.vf_net

    def train_critic(self, gradient_steps: int = 1,
                     batch_size: int = 100,
                     replay_data: Optional[ReplayBufferSamples] = None,
                     tau: float = 0.0) -> None:
        # Update optimizer learning rate
        self._update_learning_rate(self.critic.optimizer)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            if replay_data is None:
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Select action according to policy and add clipped noise
            noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(replay_data.next_observations, next_actions)
            target_q = th.min(target_q1, target_q2)
            target_q = replay_data.rewards + ((1 - replay_data.dones) * self.gamma * target_q).detach()

            # Get current Q estimates
            current_q1, current_q2 = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Update the frozen target models
            # Note: by default, for TD3, this update is done in train_actor
            # however, for CEMRL it is done here
            if tau > 0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train_actor(self, gradient_steps: int = 1,
                    batch_size: int = 100,
                    tau_actor: float = 0.005,
                    tau_critic: float = 0.005,
                    replay_data: Optional[ReplayBufferSamples] = None) -> None:
        # Update optimizer learning rate
        self._update_learning_rate(self.actor.optimizer)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            if replay_data is None:
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Compute actor loss
            actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update the frozen target models
            if tau_critic > 0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau_critic * param.data + (1 - tau_critic) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau_actor * param.data + (1 - tau_actor) * target_param.data)

    def train(self, gradient_steps: int, batch_size: int = 100, policy_delay: int = 2) -> None:

        for gradient_step in range(gradient_steps):

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            self.train_critic(replay_data=replay_data)

            # Delayed policy updates
            if gradient_step % policy_delay == 0:
                self.train_actor(replay_data=replay_data, tau_actor=self.tau, tau_critic=self.tau)

        self._n_updates += gradient_steps
        logger.logkv("n_updates", self._n_updates)


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
        policy_loss = -(advantage * log_prob).mean()

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
              reset_num_timesteps: bool = True) -> OffPolicyRLModel:

        episode_num, obs, callback = self._setup_learn(eval_env, callback, eval_freq,
                                                       n_eval_episodes, eval_log_path, reset_num_timesteps)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            rollout = self.collect_rollouts(self.env, n_episodes=self.n_episodes_rollout,
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

                if self.use_sde:
                    if self.sde_log_std_scheduler is not None:
                        # Call the scheduler
                        value = self.sde_log_std_scheduler(self._current_progress)
                        self.actor.log_std.data = th.ones_like(self.actor.log_std) * value
                    else:
                        # On-policy gradient
                        self.train_sde()

                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                self.train(gradient_steps, batch_size=self.batch_size, policy_delay=self.policy_delay)

        callback.on_training_end()

        return self

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
