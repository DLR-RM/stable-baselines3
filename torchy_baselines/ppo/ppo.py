import time
from copy import deepcopy

import gym
import torch as th
import torch.nn.functional as F
import numpy as np

from torchy_baselines.common.base_class import BaseRLModel
from torchy_baselines.common.evaluation import evaluate_policy
from torchy_baselines.common.buffers import RolloutBuffer
from torchy_baselines.common.utils import explained_variance
from torchy_baselines.common.vec_env import VecNormalize
from torchy_baselines.ppo.policies import PPOPolicy


class PPO(BaseRLModel):
    """
    Implementation of Proximal Policy Optimization (PPO) (clip version)
    Paper: https://arxiv.org/abs/1707.06347
    Code: https://github.com/openai/spinningup/
    and https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
    and stable_baselines
    """

    def __init__(self, policy, env, policy_kwargs=None, verbose=0,
                 learning_rate=3e-4, seed=0, device='auto',
                 n_optim=5, batch_size=64, n_steps=256,
                 gamma=0.99, lambda_=0.95, clip_range=0.2,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 target_kl=None, clip_range_vf=None, create_eval_env=False,
                 _init_setup_model=True):

        super(PPO, self).__init__(policy, env, PPOPolicy, policy_kwargs,
                                  verbose, device, create_eval_env=create_eval_env, support_multi_env=True)

        self.learning_rate = learning_rate
        self._seed = seed
        self.batch_size = batch_size
        self.n_optim = n_optim
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.target_kl = target_kl
        self.clip_range_vf = clip_range_vf

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        state_dim, action_dim = self.observation_space.shape[0], self.action_space.shape[0]
        # TODO: different seed for each env when n_envs > 1
        if self.n_envs == 1:
            self.seed(self._seed)

        self.rollout_buffer = RolloutBuffer(self.n_steps, state_dim, action_dim, self.device,
                                            gamma=self.gamma, lambda_=self.lambda_, n_envs=self.n_envs)
        self.policy = self.policy(self.observation_space, self.action_space,
                                  self.learning_rate, device=self.device, **self.policy_kwargs)
        self.policy = self.policy.to(self.device)

    def select_action(self, observation):
        # Normally not needed
        observation = np.array(observation)
        with th.no_grad():
            observation = th.FloatTensor(observation.reshape(1, -1)).to(self.device)
            return self.policy.actor_forward(observation, deterministic=False)

    def predict(self, observation, state=None, mask=None, deterministic=True):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        return np.clip(self.select_action(observation), self.action_space.low, self.action_space.high)

    def collect_rollouts(self, env, rollout_buffer, n_rollout_steps=256, callback=None,
                         obs=None):

        n_steps = 0
        rollout_buffer.reset()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                actions, values, log_probs = self.policy.forward(obs)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            new_obs, rewards, dones, _ = env.step(clipped_actions)

            n_steps += 1
            rollout_buffer.add(obs, actions, rewards, dones, values, log_probs)
            obs = new_obs

        rollout_buffer.compute_returns_and_advantage(values, dones=dones)

        return obs

    def train(self, n_iterations, batch_size=64):

        # TODO: replace with iterator?
        for it in range(n_iterations):
            approx_kl_divs = []
            # Sample replay buffer
            for replay_data in self.rollout_buffer.get(batch_size):
                # Unpack
                state, action, old_values, old_log_prob, advantage, return_batch = replay_data

                values, log_prob, entropy = self.policy.get_policy_stats(state, action)
                values = values.flatten()
                # Normalize advantage
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - old_log_prob)
                # clipped surrogate loss
                policy_loss_1 = advantage * ratio
                policy_loss_2 = advantage * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = old_values + th.clamp(values - old_values, -self.clip_range_vf, self.clip_range_vf)
                # Value loss using the TD(lambda_) target
                value_loss = F.mse_loss(return_batch, values_pred)


                # Entropy loss favor exploration
                entropy_loss = th.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                approx_kl_divs.append(th.mean(old_log_prob - log_prob).detach().cpu().numpy())

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print("Early stopping at step {} due to reaching max kl: {:.2f}".format(it, np.mean(approx_kl_divs)))
                break

        # print(explained_variance(self.rollout_buffer.returns.flatten().cpu().numpy(),
        #                          self.rollout_buffer.values.flatten().cpu().numpy()))

    def learn(self, total_timesteps, callback=None, log_interval=100,
              eval_env=None, eval_freq=-1, n_eval_episodes=5, tb_log_name="PPO", reset_num_timesteps=True):

        timesteps_since_eval = 0
        episode_num = 0
        evaluations = []
        start_time = time.time()
        obs = self.env.reset()
        eval_env = self._get_eval_env(eval_env)

        while self.num_timesteps < total_timesteps:

            if callback is not None:
                # Only stop training if return value is False, not when it is None.
                if callback(locals(), globals()) is False:
                    break

            obs = self.collect_rollouts(self.env, self.rollout_buffer, n_rollout_steps=self.n_steps,
                                        obs=obs)
            episode_num += 1
            self.num_timesteps += self.n_steps * self.n_envs
            timesteps_since_eval += self.n_steps * self.n_envs

            self.train(self.n_optim, batch_size=self.batch_size)

            # Evaluate agent
            if 0 < eval_freq <= timesteps_since_eval and eval_env is not None:
                timesteps_since_eval %= eval_freq
                # Sync eval env and train env when using VecNormalize
                if isinstance(self.env, VecNormalize):
                    eval_env.obs_rms = deepcopy(self.env.obs_rms)
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
