import time

import torch as th
import torch.nn.functional as F
import numpy as np

from torchy_baselines.common.base_class import BaseRLModel
from torchy_baselines.common.buffers import ReplayBuffer
from torchy_baselines.common.evaluation import evaluate_policy
from torchy_baselines.sac.policies import SACPolicy
from torchy_baselines.common.vec_env import sync_envs_normalization


class SAC(BaseRLModel):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: (SACPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)
    :param ent_coef: (str or float) Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_update_interval: (int) update the target network every `target_network_update_freq` steps.
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param target_entropy: (str or float) target entropy when learning ent_coef (ent_coef = 'auto')
    :param action_noise: (ActionNoise) the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param gamma: (float) the discount factor
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param seed: (int) Seed for the pseudo random generators
    :param device: (str or th.device) Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """
    def __init__(self, policy, env, learning_rate=3e-4, buffer_size=int(1e6),
                 learning_starts=100, batch_size=64,
                 tau=0.005, ent_coef='auto', target_update_interval=1,
                 train_freq=1, gradient_steps=1, n_episodes_rollout=-1,
                 target_entropy='auto', action_noise=None,
                 gamma=0.99, tensorboard_log=None, create_eval_env=False,
                 policy_kwargs=None, verbose=0, seed=0, device='auto',
                 _init_setup_model=True):

        super(SAC, self).__init__(policy, env, SACPolicy, policy_kwargs, verbose, device,
                                  create_eval_env=create_eval_env, seed=seed)

        self.learning_rate = learning_rate
        self.target_entropy = target_entropy
        self.log_ent_coef = None
        self.target_update_interval = target_update_interval
        self.buffer_size = buffer_size
        # In the original paper, same learning rate is used for all networks
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.n_episodes_rollout = n_episodes_rollout
        self.action_noise = action_noise
        self.gamma = gamma

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        self._setup_learning_rate()
        obs_dim, action_dim = self.observation_space.shape[0], self.action_space.shape[0]
        if self.seed is not None:
            self.set_random_seed(self.seed)

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == 'auto':
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if '_' in self.ent_coef:
                init_value = float(self.ent_coef.split('_')[1])
                assert init_value > 0., "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.learning_rate(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef = float(self.ent_coef)

        self.replay_buffer = ReplayBuffer(self.buffer_size, obs_dim, action_dim, self.device)
        self.policy = self.policy(self.observation_space, self.action_space,
                                  self.learning_rate, device=self.device, **self.policy_kwargs)
        self.policy = self.policy.to(self.device)
        self._create_aliases()

    def _create_aliases(self):
        self.actor = self.policy.actor
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
        return self.unscale_action(self.select_action(observation))

    def train(self, gradient_steps, batch_size=64):
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            obs, action_batch, next_obs, done, reward = replay_data

            # Action by the current actor for the sampled state
            action_pi, log_prob = self.actor.action_log_prob(obs)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if not isinstance(self.ent_coef, float):
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            else:
                ent_coef = self.ent_coef

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Select action according to policy
            next_action, next_log_prob = self.actor.action_log_prob(next_obs)

            with th.no_grad():
                # Compute the target Q value
                target_q1, target_q2 = self.critic_target(next_obs, next_action)
                target_q = th.min(target_q1, target_q2)
                target_q = reward + (1 - done) * self.gamma * target_q
                # td error + entropy term
                q_backup = target_q - ent_coef * next_log_prob.reshape(-1, 1)

            # Get current Q estimates
            # using action from the replay buffer
            current_q1, current_q2 = self.critic(obs, action_batch)

            # Compute critic loss
            critic_loss = 0.5 * (F.mse_loss(current_q1, q_backup) + F.mse_loss(current_q2, q_backup))

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            qf1_pi, qf2_pi = self.critic.forward(obs, action_pi)
            min_qf_pi = th.min(qf1_pi, qf2_pi)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn(self, total_timesteps, callback=None, log_interval=4,
              eval_env=None, eval_freq=-1, n_eval_episodes=5, tb_log_name="SAC",
              reset_num_timesteps=True):

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

            self.num_timesteps += episode_timesteps
            episode_num += n_episodes
            timesteps_since_eval += episode_timesteps
            self._update_current_progress(self.num_timesteps, total_timesteps)

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                if self.verbose > 1:
                    print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(
                        self.num_timesteps, episode_num, episode_timesteps, episode_reward))
                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else episode_timesteps

                self.train(gradient_steps, batch_size=self.batch_size)

            # Evaluate episode
            if 0 < eval_freq <= timesteps_since_eval and eval_env is not None:
                timesteps_since_eval %= eval_freq
                sync_envs_normalization(self.env, eval_env)
                mean_reward, _ = evaluate_policy(self, eval_env, n_eval_episodes)
                evaluations.append(mean_reward)
                if self.verbose > 0:
                    print("Eval num_timesteps={}, mean_reward={:.2f}".format(self.num_timesteps, evaluations[-1]))
                    print("FPS: {:.2f}".format(self.num_timesteps / (time.time() - self.start_time)))

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
