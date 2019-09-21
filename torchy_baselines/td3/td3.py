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
    Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
    Paper: https://arxiv.org/abs/1802.09477
    Code: https://github.com/sfujim/TD3
    """

    def __init__(self, policy, env, policy_kwargs=None, verbose=0,
                 buffer_size=int(1e6), learning_rate=1e-3, seed=0, device='auto',
                 action_noise_std=0.1, start_timesteps=100, policy_freq=2,
                 batch_size=100, create_eval_env=False,
                 _init_setup_model=True):

        super(TD3, self).__init__(policy, env, TD3Policy, policy_kwargs, verbose, device,
                                  create_eval_env=create_eval_env)

        self.max_action = np.abs(self.action_space.high)
        self.action_noise_std = action_noise_std
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.start_timesteps = start_timesteps
        self._seed = seed
        self.policy_freq = policy_freq
        self.batch_size = batch_size

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        state_dim, action_dim = self.observation_space.shape[0], self.action_space.shape[0]
        self.seed(self._seed)
        self.replay_buffer = ReplayBuffer(self.buffer_size, state_dim, action_dim, self.device)
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
            return self.actor(observation).cpu().data.numpy()

    def predict(self, observation, state=None, mask=None, deterministic=True):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        return self.max_action * self.select_action(observation)

    def train_critic(self, n_iterations=1, batch_size=100, discount=0.99,
                     policy_noise=0.2, noise_clip=0.5, replay_data=None, tau=0.0):

        for it in range(n_iterations):
            # Sample replay buffer
            if replay_data is None:
                state, action, next_state, done, reward = self.replay_buffer.sample(batch_size)
            else:
                state, action, next_state, done, reward = replay_data

            # Select action according to policy and add clipped noise
            noise = action.clone().data.normal_(0, policy_noise)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = th.min(target_q1, target_q2)
            target_q = reward + ((1 - done) * discount * target_q).detach()

            # Get current Q estimates
            current_q1, current_q2 = self.critic(state, action)

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

    def train(self, n_iterations, batch_size=100, policy_freq=2):

        for it in range(n_iterations):

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)
            self.train_critic(replay_data=replay_data)

            # Delayed policy updates
            if it % policy_freq == 0:
                self.train_actor(replay_data=replay_data)

    def learn(self, total_timesteps, callback=None, log_interval=100,
              eval_env=None, eval_freq=-1, n_eval_episodes=5, tb_log_name="TD3", reset_num_timesteps=True):

        timesteps_since_eval = 0
        episode_num = 0
        evaluations = []
        start_time = time.time()
        eval_env = self._get_eval_env(eval_env)

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
                                                                      replay_buffer=self.replay_buffer)
            episode_num += 1
            self.num_timesteps += episode_timesteps
            timesteps_since_eval += episode_timesteps

            if self.num_timesteps > 0:
                if self.verbose > 1:
                    print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(
                        self.num_timesteps, episode_num, episode_timesteps, episode_reward))
                self.train(episode_timesteps, batch_size=self.batch_size, policy_freq=self.policy_freq)

            # Evaluate episode
            if 0 < eval_freq <= timesteps_since_eval and eval_env is not None:
                timesteps_since_eval %= eval_freq
                mean_reward, _ = evaluate_policy(self, eval_env, n_eval_episodes)
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
        self._create_aliases()
