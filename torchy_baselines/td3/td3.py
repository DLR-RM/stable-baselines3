import time

import torch as th
import torch.nn.functional as F
import numpy as np

from torchy_baselines.common.base_class import BaseRLModel
from torchy_baselines.common.buffers import ReplayBuffer
from torchy_baselines.common.evaluation import evaluate_policy
from torchy_baselines.td3.policies import TD3Policy


class TD3(BaseRLModel):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: (TD3Policy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param buffer_size: (int) size of the replay buffer
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values and Actor networks)
        it can be a function of the current progress (from 1 to 0)
    :param policy_delay: (int) Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param gamma: (float) the discount factor
    :param batch_size: (int) Minibatch size for each gradient update
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param tau: (float) the soft update coefficient ("polyak update" of the target networks, between 0 and 1)
    :param action_noise: (ActionNoise) the action noise type. Cf common.noise for the different action noise type.
    :param target_policy_noise: (float) Standard deviation of gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: (float) Limit for absolute value of target policy smoothing noise.
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param seed: (int) Seed for the pseudo random generators
    :param device: (str or th.device) Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(self, policy, env, buffer_size=int(1e6), learning_rate=1e-3,
                 policy_delay=2, learning_starts=100, gamma=0.99, batch_size=100,
                 train_freq=-1, gradient_steps=-1, n_episodes_rollout=1,
                 tau=0.005, action_noise=None, target_policy_noise=0.2, target_noise_clip=0.5,
                 tensorboard_log=None, create_eval_env=False, policy_kwargs=None, verbose=0,
                 seed=0, device='auto', _init_setup_model=True):

        super(TD3, self).__init__(policy, env, TD3Policy, policy_kwargs, verbose, device,
                                  create_eval_env=create_eval_env, seed=seed)

        self.buffer_size = buffer_size
        # TODO: accept callables
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

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        self._setup_learning_rate()
        obs_dim, action_dim = self.observation_space.shape[0], self.action_space.shape[0]
        self.set_random_seed(self.seed)
        self.replay_buffer = ReplayBuffer(self.buffer_size, obs_dim, action_dim, self.device)
        self.policy = self.policy(self.observation_space, self.action_space,
                                  self.learning_rate, device=self.device, **self.policy_kwargs)
        self.policy = self.policy.to(self.device)
        self._create_aliases()

    def _create_aliases(self):
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def select_action(self, observation):
        # Normally not needed
        observation = np.array(observation)
        with th.no_grad():
            observation = th.FloatTensor(observation.reshape(1, -1)).to(self.device)
            return self.actor(observation).cpu().numpy()

    def predict(self, observation, state=None, mask=None, deterministic=True):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        return self.unscale_action(self.select_action(observation))

    def train_critic(self, gradient_steps=1, batch_size=100, replay_data=None, tau=0.0):
        # Update optimizer learning rate
        self._update_learning_rate(self.critic.optimizer)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            if replay_data is None:
                obs, action, next_obs, done, reward = self.replay_buffer.sample(batch_size)
            else:
                obs, action, next_obs, done, reward = replay_data

            # Select action according to policy and add clipped noise
            noise = action.clone().data.normal_(0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_action = (self.actor_target(next_obs) + noise).clamp(-1, 1)

            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = th.min(target_q1, target_q2)
            target_q = reward + ((1 - done) * self.gamma * target_q).detach()

            # Get current Q estimates
            current_q1, current_q2 = self.critic(obs, action)

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

    def train_actor(self, gradient_steps: object = 1, batch_size: object = 100, tau_actor: object = 0.005,
                    tau_critic: object = 0.005,
                    replay_data: object = None) -> object:
        # Update optimizer learning rate
        self._update_learning_rate(self.actor.optimizer)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            if replay_data is None:
                obs, _, next_obs, done, reward = self.replay_buffer.sample(batch_size)
            else:
                obs, _, next_obs, done, reward = replay_data

            # Compute actor loss
            actor_loss = -self.critic.q1_forward(obs, self.actor(obs)).mean()

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

    def train(self, gradient_steps, batch_size=100, policy_delay=2):

        for gradient_step in range(gradient_steps):

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)
            self.train_critic(replay_data=replay_data)

            # Delayed policy updates
            if gradient_step % policy_delay == 0:
                self.train_actor(replay_data=replay_data, tau_actor=self.tau, tau_critic=self.tau)

    def learn(self, total_timesteps, callback=None, log_interval=4,
              eval_env=None, eval_freq=-1, n_eval_episodes=5, tb_log_name="TD3", reset_num_timesteps=True):

        timesteps_since_eval, episode_num, evaluations, obs, eval_env = self._setup_learn(eval_env)

        while self.num_timesteps < total_timesteps:

            if callback is not None:
                # Only stop training if return value is False, not when it is None.
                if callback(locals(), globals()) is False:
                    break

            rollout = self.collect_rollouts(self.env, n_episodes=self.n_episodes_rollout,
                                            n_steps=self.train_freq, action_noise=self.action_noise,
                                            deterministic=False, callback=None,
                                            learning_starts=self.learning_starts,
                                            num_timesteps=self.num_timesteps,
                                            replay_buffer=self.replay_buffer,
                                            obs=obs, episode_num=episode_num,
                                            log_interval=log_interval)
            # Unpack
            episode_reward, episode_timesteps, n_episodes, obs = rollout

            episode_num += n_episodes
            self.num_timesteps += episode_timesteps
            timesteps_since_eval += episode_timesteps
            self._update_current_progress(self.num_timesteps, total_timesteps)

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                if self.verbose > 1:
                    print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(
                        self.num_timesteps, episode_num, episode_timesteps, episode_reward))

                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else episode_timesteps
                self.train(gradient_steps, batch_size=self.batch_size, policy_delay=self.policy_delay)

            # Evaluate episode
            if 0 < eval_freq <= timesteps_since_eval and eval_env is not None:
                timesteps_since_eval %= eval_freq
                mean_reward, _ = evaluate_policy(self, eval_env, n_eval_episodes)
                evaluations.append(mean_reward)
                if self.verbose > 0:
                    print("Eval num_timesteps={}, mean_reward={:.2f}".format(self.num_timesteps, evaluations[-1]))
                    print("FPS: {:.2f}".format(self.num_timesteps / (time.time() - self.start_time)))

        return self

    def get_opt_parameters(self):
        """
        returns a dict of all the optimizers and their parameters

        :return: (Dict) of optimizer names and their state_dict 
        """
        return {"actor": self.actor.optimizer.state_dict(), "critic": self.critic.optimizer.state_dict()}

    def load_parameters(self, load_dict, opt_params):
        """
        Load model parameters and optimizer parameters from a dictionary
        Dictionary should be of shape torch model.state_dict()
        This does not load agent's hyper-parameters.

        :param load_dict: (dict) dict of parameters from model.state_dict()
        :param opt_params: (dict of dicts) dict of optimizer state_dicts should be handled in child_class
        """
        self.actor.optimizer.load_state_dict(opt_params["actor"])
        self.critic.optimizer.load_state_dict(opt_params["critic"])
        self.policy.load_state_dict(load_dict)
