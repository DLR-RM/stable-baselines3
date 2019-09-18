import time

import torch as th
import torch.nn.functional as F
import numpy as np

from torchy_baselines.common.base_class import BaseRLModel
from torchy_baselines.common.utils import set_random_seed
from torchy_baselines.common.evaluation import evaluate_policy
from torchy_baselines.ppo.policies import ActorCriticPolicy
from torchy_baselines.common.replay_buffer import ReplayBuffer


class PPO(BaseRLModel):
    """
    Implementation of Proximal Policy Optimization (PPO) (clip version)
    Paper: https://arxiv.org/abs/1707.06347
    Code: https://github.com/openai/spinningup/
    """

    def __init__(self, policy, env, policy_kwargs=None, verbose=0,
                 learning_rate=1e-3, seed=0, device='auto',
                 n_optim=5, batch_size=100, n_steps=256,
                 gamma=0.99, lambda_=0.95,
                _init_setup_model=True):

        super(PPO, self).__init__(policy, env, ActorCriticPolicy, policy_kwargs, verbose, device)

        self.max_action = np.abs(self.action_space.high)
        self.learning_rate = learning_rate
        self._seed = seed
        self.batch_size = batch_size
        self.n_optim = n_optim
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_ = lambda_
        self.buffer_rollouts = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        state_dim, action_dim = self.observation_space.shape[0], self.action_space.shape[0]
        self.seed(self._seed)

        self.policy = self.policy(self.observation_space, self.action_space,
                                  self.learning_rate, device=self.device, **self.policy_kwargs)


    def select_action(self, observation):
        # Normally not needed
        observation = np.array(observation)
        with th.no_grad():
            observation = th.FloatTensor(observation.reshape(1, -1)).to(self.device)
            return self.policy.actor_forward(observation).cpu().data.numpy().flatten()

    def predict(self, observation, state=None, mask=None, deterministic=True):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        return np.clip(self.select_action(observation), -self.max_action, self.max_action)


    def train_actor(self, n_iterations=1, batch_size=100, tau_actor=0.005, tau_critic=0.005, replay_data=None):

        for it in range(n_iterations):
            # Sample replay buffer
            if replay_data is None:
                state, action, next_state, done, reward = self.replay_buffer.sample(batch_size)
            else:
                state, action, next_state, done, reward = replay_data

            # Compute actor loss
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()

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

    def train(self, n_iterations, batch_size=100, discount=0.99,
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(n_iterations):

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)
            self.train_critic(replay_data=replay_data)

            # Delayed policy updates
            if it % policy_freq == 0:
                self.train_actor(replay_data=replay_data)

    def learn(self, total_timesteps, callback=None, log_interval=100,
              eval_freq=-1, n_eval_episodes=5, tb_log_name="TD3", reset_num_timesteps=True):

        timesteps_since_eval = 0
        episode_num = 0
        evaluations = []
        start_time = time.time()

        while self.num_timesteps < total_timesteps:

            if callback is not None:
                # Only stop training if return value is False, not when it is None.
                if callback(locals(), globals()) is False:
                    break

            episode_reward, episode_timesteps = self.collect_rollouts(self.env, n_episodes=1,
                                                                      action_noise_std=self.action_noise_std,
                                                                      deterministic=False, callback=None,
                                                                      start_timesteps=self.start_timesteps,
                                                                      num_timesteps=self.num_timesteps,
                                                                      replay_buffer=self.buffer_rollouts)
            episode_num += 1
            self.num_timesteps += episode_timesteps
            timesteps_since_eval += episode_timesteps

            if self.num_timesteps > 0:
                if self.verbose > 1:
                    print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(
                          self.num_timesteps, episode_num, episode_timesteps, episode_reward))
                self.train(episode_timesteps, batch_size=self.batch_size, policy_freq=self.policy_freq)

            # Evaluate episode
            if 0 < eval_freq <= timesteps_since_eval:
                timesteps_since_eval %= eval_freq
                mean_reward, _ = evaluate_policy(self, self.env, n_eval_episodes)
                evaluations.append(mean_reward)
                if self.verbose > 0:
                    print("Eval num_timesteps={}, mean_reward={:.2f}".format(self.num_timesteps, evaluations[-1]))
                    print("FPS: {:.2f}".format(self.num_timesteps / (time.time() - start_time)))

        return self

    def save(self, path):
        if not path.endswith('.pth'):
            path += '.pth'
        th.save(self.policy.state_dict(), path)

    def load(self, path, env=None, **_kwargs):
        if not path.endswith('.pth'):
            path += '.pth'
        if env is not None:
            pass
        self.policy.load_state_dict(th.load(path))


class PPOBuffer(ReplayBuffer):
    """docstring for PPOBuffer."""

    def __init__(self, buffer_size, state_dim, action_dim, device='cpu',
                lambda=0.95):
        super(PPOBuffer, self).__init__(buffer_size, state_dim, action_dim, device)

        self.returns = th.zeros(self.buffer_size, 1)
        self.values = th.zeros(self.buffer_size, 1)
        self.log_probs = th.zeros(self.buffer_size, 1)
        self.advantages = th.zeros(self.buffer_size, 1)

    def compute_gae(self):
        """
        From https://github.com/openai/spinningup/blob/master/spinup/algos/ppo/ppo.py
        """
        path_slice = slice(self.path_start_idx, self.pos)
        rews = np.append(self.rewards[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.advantages[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.pos
