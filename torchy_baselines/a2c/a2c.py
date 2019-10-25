from gym import spaces
import torch as th
import torch.nn.functional as F

from torchy_baselines.common.utils import explained_variance
from torchy_baselines.ppo.ppo import PPO
from torchy_baselines.ppo.policies import PPOPolicy


class A2C(PPO):
    """
    Advantage Actor Critic (A2C)

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

    :param policy: (PPOPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param rms_prop_eps: (float) RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: (bool) Whether to use RMSprop (default) or Adam as optimizer
    :param normalize_advantage: (bool) Whether to normalize or not the advantage
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

    def __init__(self, policy, env, learning_rate=7e-4,
                 n_steps=5, gamma=0.99, gae_lambda=1.0,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 rms_prop_eps=1e-5, use_rms_prop=True,
                 normalize_advantage=False, tensorboard_log=None, create_eval_env=False,
                 policy_kwargs=None, verbose=0, seed=0, device='auto',
                 _init_setup_model=True):

        super(A2C, self).__init__(policy, env, learning_rate=learning_rate,
                                  n_steps=n_steps, batch_size=n_steps, n_epochs=1,
                                  gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef,
                                  vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                                  tensorboard_log=tensorboard_log, policy_kwargs=policy_kwargs,
                                  verbose=verbose, device=device, create_eval_env=create_eval_env,
                                  seed=seed, _init_setup_model=False)

        self.normalize_advantage = normalize_advantage
        self.rms_prop_eps = rms_prop_eps
        self.use_rms_prop = use_rms_prop

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        super(A2C, self)._setup_model()
        if self.use_rms_prop:
            self.policy.optimizer = th.optim.RMSprop(self.policy.parameters(),
                                                     lr=self.learning_rate, alpha=0.99,
                                                     eps=self.rms_prop_eps, weight_decay=0)

    def train(self, gradient_steps, batch_size=64):

        for gradient_step in range(gradient_steps):
            # approx_kl_divs = []
            # Sample replay buffer
            for replay_data in self.rollout_buffer.get(batch_size):
                # Unpack
                obs, action, _, _, advantage, return_batch = replay_data

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action for float to long
                    action = action.long().flatten()

                values, log_prob, entropy = self.policy.get_policy_stats(obs, action)
                values = values.flatten()
                # Normalize advantage (not present in the original implementation)
                if self.normalize_advantage:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                policy_loss = -(advantage * log_prob).mean()

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(return_batch, values)

                # Entropy loss favor exploration
                entropy_loss = th.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                # approx_kl_divs.append(th.mean(old_log_prob - log_prob).detach().cpu().numpy())

        # print(explained_variance(self.rollout_buffer.returns.flatten().cpu().numpy(),
        #                          self.rollout_buffer.values.flatten().cpu().numpy()))

    def learn(self, total_timesteps, callback=None, log_interval=100,
              eval_env=None, eval_freq=-1, n_eval_episodes=5, tb_log_name="A2C", reset_num_timesteps=True):

        return super(A2C, self).learn(total_timesteps=total_timesteps, callback=callback, log_interval=log_interval,
                  eval_env=eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
                  tb_log_name=tb_log_name, reset_num_timesteps=reset_num_timesteps)
