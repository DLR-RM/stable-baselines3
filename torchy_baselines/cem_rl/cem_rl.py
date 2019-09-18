import time

import torch as th
import torch.nn.functional as F
import numpy as np

from torchy_baselines.td3.td3 import TD3
from torchy_baselines.common.evaluation import evaluate_policy
from torchy_baselines.cem_rl.cem import CEM


class CEMRL(TD3):
    """
    Implementation of CEM-RL

    Paper: https://arxiv.org/abs/1810.01222
    Code: https://github.com/apourchot/CEM-RL
    """
    def __init__(self, policy, env, policy_kwargs=None, verbose=0,
                 sigma_init=1e-3, pop_size=10, damp=1e-3, damp_limit=1e-5,
                 elitism=False, n_grad=5, policy_freq=2, batch_size=100,
                 buffer_size=int(1e6), learning_rate=1e-3, seed=0, device='auto',
                 action_noise_std=0.0, start_timesteps=100, update_style='original',
                 _init_setup_model=True):

        super(CEMRL, self).__init__(policy, env, policy_kwargs, verbose,
                                    buffer_size, learning_rate, seed, device,
                                    action_noise_std, start_timesteps,
                                    policy_freq=policy_freq, batch_size=batch_size,
                                    _init_setup_model=False)

        self.es = None
        self.sigma_init = sigma_init
        self.pop_size = pop_size
        self.damp = damp
        self.damp_limit = damp_limit
        self.elitism = elitism
        self.n_grad = n_grad
        self.es_params = None
        self.update_style = update_style
        self.fitnesses = []

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self, seed=None):
        super(CEMRL, self)._setup_model()
        params_vector = self.actor.parameters_to_vector()
        self.es = CEM(len(params_vector), mu_init=params_vector,
                      sigma_init=self.sigma_init, damp=self.damp, damp_limit=self.damp_limit,
                      pop_size=self.pop_size, antithetic=not self.pop_size % 2, parents=self.pop_size // 2,
                      elitism=self.elitism)

    def learn(self, total_timesteps, callback=None, log_interval=100,
              eval_freq=-1, n_eval_episodes=5, tb_log_name="CEMRL", reset_num_timesteps=True):

        timesteps_since_eval, actor_steps = 0, 0
        episode_num = 0
        evaluations = []
        start_time = time.time()

        while self.num_timesteps < total_timesteps:

            self.fitnesses = []
            self.es_params = self.es.ask(self.pop_size)

            if callback is not None:
                # Only stop training if return value is False, not when it is None.
                if callback(locals(), globals()) is False:
                    break

            if self.num_timesteps > 0:
                # self.train(episode_timesteps)
                # Gradient steps for half of the population
                for i in range(self.n_grad):
                    # set params
                    self.actor.load_from_vector(self.es_params[i])
                    self.actor_target.load_from_vector(self.es_params[i])
                    self.actor.optimizer = th.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

                    # In the paper: 2 * actor_steps // self.n_grad
                    # In the original implementation: actor_steps // self.n_grad
                    # Difference with TD3 implementation:
                    # the target critic is updated in the train_critic()
                    # instead of the train_actor() and no policy delay
                    # Issue with this update style: the bigger the population, the slower the code
                    if self.update_style == 'original':
                        self.train_critic(actor_steps // self.n_grad, tau=0.005)
                        self.train_actor(actor_steps, tau_critic=0.0)
                    elif self.update_style == 'original_td3':
                        self.train_critic(actor_steps // self.n_grad, tau=0.0)
                        self.train_actor(actor_steps)
                    else:
                        # Closer to td3: with policy delay
                        if self.update_style == 'td3_like':
                            n_training_steps = actor_steps
                        else:
                            # scales with a bigger population
                            # but less training steps per agent
                            n_training_steps == 2 * (actor_steps // self.n_grad)
                        for it in range(n_training_steps):
                            # Sample replay buffer
                            replay_data = self.replay_buffer.sample(self.batch_size)
                            self.train_critic(replay_data=replay_data)

                            # Delayed policy updates
                            if it % self.policy_freq == 0:
                                self.train_actor(replay_data=replay_data)

                    # Get the params back in the population
                    self.es_params[i] = self.actor.parameters_to_vector()

            # Evaluate episode
            if 0 < eval_freq <= timesteps_since_eval:
                timesteps_since_eval %= eval_freq

                self.actor.load_from_vector(self.es.mu)

                mean_reward, _ = evaluate_policy(self, self.env, n_eval_episodes)
                evaluations.append(mean_reward)

                if self.verbose > 0:
                    print("Eval num_timesteps={}, mean_reward={:.2f}".format(self.num_timesteps, evaluations[-1]))
                    print("FPS: {:.2f}".format(self.num_timesteps / (time.time() - start_time)))

            actor_steps = 0
            # evaluate all actors
            for params in self.es_params:

                self.actor.load_from_vector(params)

                episode_reward, episode_timesteps = self.collect_rollouts(self.env, n_episodes=1,
                                                                          action_noise_std=self.action_noise_std,
                                                                          deterministic=False, callback=None,
                                                                          start_timesteps=self.start_timesteps,
                                                                          num_timesteps=self.num_timesteps,
                                                                          replay_buffer=self.replay_buffer)
                episode_num += 1
                self.num_timesteps += episode_timesteps
                timesteps_since_eval += episode_timesteps
                actor_steps += episode_timesteps
                self.fitnesses.append(episode_reward)

                if self.verbose > 1:
                    print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(
                        self.num_timesteps, episode_num, episode_timesteps, episode_reward))


            self.es.tell(self.es_params, self.fitnesses)
            timesteps_since_eval += actor_steps
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
