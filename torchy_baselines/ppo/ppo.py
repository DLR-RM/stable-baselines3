import time

import torch as th
import torch.nn.functional as F
import numpy as np

from torchy_baselines.common.base_class import BaseRLModel
from torchy_baselines.common.evaluation import evaluate_policy
from torchy_baselines.ppo.policies import PPOPolicy
from torchy_baselines.common.buffers import RolloutBuffer


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
                 ent_coef=0.01, vf_coef=0.5,
                 _init_setup_model=True):

        super(PPO, self).__init__(policy, env, PPOPolicy, policy_kwargs, verbose, device)

        self.max_action = np.abs(self.action_space.high)
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
        self.rollout_buffer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        state_dim, action_dim = self.observation_space.shape[0], self.action_space.shape[0]
        self.seed(self._seed)

        self.rollout_buffer = RolloutBuffer(self.n_steps, state_dim, action_dim, self.device,
                                            gamma=self.gamma, lambda_=self.lambda_)
        self.policy = self.policy(self.observation_space, self.action_space,
                                  self.learning_rate, device=self.device, **self.policy_kwargs)

    def select_action(self, observation):
        # Normally not needed
        observation = np.array(observation)
        with th.no_grad():
            observation = th.FloatTensor(observation.reshape(1, -1)).to(self.device)
            return self.policy.actor_forward(observation, deterministic=False).flatten()

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

    def collect_rollouts(self, env, rollout_buffer, n_rollout_steps=256, callback=None,
                         obs=None):

        n_steps = 0
        done = obs is None
        rollout_buffer.reset()

        while n_steps < n_rollout_steps:
            # Reset environment
            if done:
                obs = env.reset()

            # No grad ok?
            with th.no_grad():
                action, value, log_prob = self.policy.forward(obs)
            action = action[0].cpu().numpy()

            # Rescale and perform action
            obs, reward, done, _ = env.step(np.clip(action, -self.max_action, self.max_action))

            n_steps += 1
            rollout_buffer.add(obs, action, reward, float(done), value, log_prob)

            if done:
                value = 0.0
                obs = None

            rollout_buffer.finish_path(last_value=value)

        return obs

    def train(self, n_iterations, batch_size=64):

        # TODO: replace with iterator?
        for it in range(n_iterations):
            # Sample replay buffer
            for replay_data in self.rollout_buffer.get(batch_size):
                # Unpack
                state, action, old_log_prob, advantage, return_batch = replay_data

                # _, values, log_prob = self.policy.forward(state)
                values, log_prob, entropy = self.policy.get_policy_stats(state, action)

                # Normalize advantage
                # advs = returns - values
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                ratio = th.exp(log_prob - old_log_prob)
                policy_loss_1 = advantage * ratio
                policy_loss_2 = advantage * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                # value_loss = th.mean((return_batch - value)**2)
                value_loss = F.mse_loss(return_batch, values)
                entropy_loss = th.mean(entropy)
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                # loss = policy_loss
                # TODO: check kl div
                # self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                # approx_kl_div = th.mean(old_log_prob - log_prob)
                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # TODO: clip grad norm?
                # nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

    def learn(self, total_timesteps, callback=None, log_interval=100,
              eval_freq=-1, n_eval_episodes=5, tb_log_name="PPO", reset_num_timesteps=True):

        timesteps_since_eval = 0
        episode_num = 0
        evaluations = []
        start_time = time.time()
        obs = None

        while self.num_timesteps < total_timesteps:

            if callback is not None:
                # Only stop training if return value is False, not when it is None.
                if callback(locals(), globals()) is False:
                    break

            # TODO: avoid reset using obs=obs and test env
            obs = self.collect_rollouts(self.env, self.rollout_buffer, n_rollout_steps=self.n_steps,
                                        obs=None)
            episode_num += 1
            self.num_timesteps += self.n_steps
            timesteps_since_eval += self.n_steps

            self.train(self.n_optim, batch_size=self.batch_size)

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
