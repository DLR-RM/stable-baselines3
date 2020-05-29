from typing import List, Tuple, Type, Union, Callable, Optional, Dict, Any

import torch as th
import torch.nn.functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.base_class import OffPolicyRLModel
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.dqn.policies import DQNPolicy


class DQN(OffPolicyRLModel):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default parameters are taken from the nature paper

    :param policy: (PPOPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) The learning rate, it can be a function
        of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param batch_size: (int) Minibatch size for each gradient update
    :param gamma: (float) Discount factor
    :param train_freq: (int) Update the model every ``train_freq`` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)
    :param target_update_interval: (int) update the target network every ``target_update_interval`` steps.
    :param exploration_fraction: (float) fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: (float) initial value of random action probability
    :param exploration_final_eps: (float) final value of random action probability
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: (int) Seed for the pseudo random generators
    :param device: (str or th.device) Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(self, policy: Union[str, Type[DQNPolicy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Callable] = 25e-5,
                 buffer_size: int = 1000000,
                 learning_starts: int = 50000,
                 batch_size: Optional[int] = 32,
                 gamma: float = 0.99,
                 train_freq: int = 1,
                 gradient_steps: int = 1,
                 tau: float = 1.0,
                 target_update_interval: int = 10000,
                 exploration_fraction: float = 1,
                 exploration_initial_eps: float = 1.0,
                 exploration_final_eps: float = 0.1,
                 max_grad_norm: float = 10,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = 'auto',
                 _init_setup_model: bool = True):

        super(DQN, self).__init__(policy, env, DQNPolicy, learning_rate,
                                  buffer_size, learning_starts, batch_size,
                                  policy_kwargs, verbose, device,
                                  create_eval_env=create_eval_env,
                                  seed=seed, sde_support=False)

        assert train_freq > 0, "``train_freq`` must be positive"
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.tensorboard_log = tensorboard_log

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(DQN, self)._setup_model()
        self._create_aliases()
        self._setup_exploration_schedule()

    def _setup_exploration_schedule(self) -> None:
        """
        Generate a exploration schedule used for updating the exploration probability
        """
        self.exploration_schedule = get_linear_fn(self.exploration_initial_eps, self.exploration_final_eps,
                                                  self.exploration_fraction)

    def _update_exploration(self) -> None:
        """
        Update the policy exploration probability using the current exploration rate schedule
        and the current progress (from 1 to 0).
        """
        # Log the current exploration probability
        logger.logkv("exploration rate", self.exploration_schedule(self._current_progress))

        self.policy.update_exploration_rate(self.exploration_schedule(self._current_progress))

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate and exploration probability according to schedule
        self._update_learning_rate(self.policy.optimizer)
        self._update_exploration()

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the target Q values
                target_q = self.q_net_target(replay_data.next_observations)
                target_q, _ = target_q.max(dim=1)
                target_q = target_q.reshape(-1, 1)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates
            current_q = self.q_net(replay_data.observations)

            # Gather q_values of our actions
            current_q = th.gather(current_q, dim=1, index=replay_data.actions.long())

            # Compute q loss
            loss = F.mse_loss(current_q, target_q)

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Update target networks
            if self._n_updates % self.target_update_interval == 0:
                for param, target_param in zip(self.q_net.parameters(), self.q_net_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self._n_updates += gradient_steps
        logger.logkv("n_updates", self._n_updates)

    def learn(self,
              total_timesteps: int,
              callback: MaybeCallback = None,
              log_interval: int = 4,
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "DQN",
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> OffPolicyRLModel:

        callback = self._setup_learn(eval_env, callback, eval_freq,
                                     n_eval_episodes, eval_log_path, reset_num_timesteps)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            rollout = self.collect_rollouts(self.env,
                                            n_steps=self.train_freq, n_episodes=-1, action_noise=None,
                                            callback=callback,
                                            learning_starts=self.learning_starts,
                                            replay_buffer=self.replay_buffer,
                                            log_interval=log_interval)

            if rollout.continue_training is False:
                break

            self._update_current_progress(self.num_timesteps, total_timesteps)

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                self.train(batch_size=self.batch_size, gradient_steps=self.gradient_steps)

        callback.on_training_end()

        return self

    def excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded by default
        when saving the model.

        :return: (List[str]) List of parameters that should be excluded from save
        """
        # Exclude aliases
        return super(DQN, self).excluded_save_params() + ["q_net", "q_net_target"]

    def get_torch_variables(self) -> Tuple[List[str], List[str]]:
        """
        cf base class
        """
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
