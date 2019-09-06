import sys
import time

import torch as th
import torch.nn.functional as F
import numpy as np

from torchy_baselines.common.base_class import BaseRLModel
from torchy_baselines.common.replay_buffer import ReplayBuffer
from torchy_baselines.common.utils import set_random_seed
from torchy_baselines.common.evaluation import evaluate_policy
from torchy_baselines.td3.policies import TD3Policy


class TD3(BaseRLModel):
    """
    Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
    Paper: https://arxiv.org/abs/1802.09477
    Code: https://github.com/sfujim/TD3
    """

    def __init__(self, policy, env, policy_kwargs=None, verbose=0,
                 buffer_size=int(1e6), learning_rate=1e-3, seed=0, device='cpu',
                 action_noise_std=0.1, start_timesteps=100, _init_setup_model=True):

        super(TD3, self).__init__(policy, env, TD3Policy, policy_kwargs, verbose)

        self.max_action = np.abs(self.action_space.high)
        self.replay_buffer = None
        self.device = device
        self.action_noise_std = action_noise_std
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.start_timesteps = start_timesteps
        self.seed = 0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self, seed=None):
        state_dim, action_dim = self.observation_space.shape[0], self.action_space.shape[0]
        set_random_seed(self.seed, using_cuda=self.device != 'cpu')

        if self.env is not None:
            self.env.seed(self.seed)

        self.replay_buffer = ReplayBuffer(self.buffer_size, state_dim, action_dim, self.device)
        self.policy = self.policy(self.observation_space, self.action_space,
                                  self.learning_rate, device=self.device, **self.policy_kwargs)
        self._create_aliases()

    def _create_aliases(self):
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def select_action(self, observation):
        with th.no_grad():
            observation = th.FloatTensor(observation.reshape(1, -1)).to(self.device)
            return self.actor(observation).cpu().data.numpy().flatten()

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

    def train(self, n_iterations, batch_size=100, discount=0.99,
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(n_iterations):

            # Sample replay buffer
            state, action, next_state, done, reward = self.replay_buffer.sample(batch_size)

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

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def learn(self, total_timesteps, callback=None, log_interval=100,
              eval_freq=-1, n_eval_episodes=5, tb_log_name="TD3", reset_num_timesteps=True):

        timesteps_since_eval = 0
        episode_num = 0
        done = True
        evaluations = []
        start_time = time.time()

        while self.num_timesteps < total_timesteps:

            if callback is not None:
                # Only stop training if return value is False, not when it is None.
                if callback(locals(), globals()) is False:
                    break

            if done:
                if self.num_timesteps > 0:
                    if self.verbose > 1:
                        print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(
                              self.num_timesteps, episode_num, episode_timesteps, episode_reward))
                    self.train(episode_timesteps)

                # Evaluate episode
                if eval_freq > 0 and timesteps_since_eval >= eval_freq:
                    timesteps_since_eval %= eval_freq
                    evaluations.append(evaluate_policy(self, self.env, n_eval_episodes))
                    if self.verbose > 0:
                        print("Eval num_timesteps={}, mean_reward={:.2f}".format(self.num_timesteps, evaluations[-1]))
                        print("FPS: {:.2f}".format(self.num_timesteps / (time.time() - start_time)))
                        sys.stdout.flush()

                # Reset environment
                obs = self.env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Select action randomly or according to policy
            if self.num_timesteps < self.start_timesteps:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(np.array(obs))

            if self.action_noise_std > 0:
                # NOTE: in the original implementation, the noise is applied to the unscaled action
                action_noise = np.random.normal(0, self.action_noise_std, size=self.action_space.shape[0])
                action = (action + action_noise).clip(-1, 1)

            # Rescale and perform action
            new_obs, reward, done, _ = self.env.step(self.max_action * action)

            if hasattr(self.env, '_max_episode_steps'):
                done_bool = 0 if episode_timesteps + 1 == self.env._max_episode_steps else float(done)
            else:
                done_bool = float(done)

            episode_reward += reward

            # Store data in replay buffer
            # self.replay_buffer.add(state, next_state, action, reward, done)
            self.replay_buffer.add(obs, new_obs, action, reward, done_bool)

            obs = new_obs

            episode_timesteps += 1
            self.num_timesteps += 1
            timesteps_since_eval += 1
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
