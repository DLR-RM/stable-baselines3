from typing import List, Tuple, Type, Union, Callable, Optional, Dict, Any

import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.dqn.policies import DQNPolicy


class DQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: (DQNPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) The learning rate, it can be a function
        of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: (float) the discount factor
    :param train_freq: (int) Update the model every ``train_freq`` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param n_episodes_rollout: (int) Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``
    :param optimize_memory_usage: (bool) Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: (int) update the target network every ``target_update_interval``
        environment steps.
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
                 learning_rate: Union[float, Callable] = 1e-4,
                 buffer_size: int = 1000000,
                 learning_starts: int = 50000,
                 batch_size: Optional[int] = 32,
                 tau: float = 1.0,
                 gamma: float = 0.99,
                 train_freq: int = 4,
                 gradient_steps: int = 1,
                 n_episodes_rollout: int = -1,
                 optimize_memory_usage: bool = False,
                 target_update_interval: int = 10000,
                 exploration_fraction: float = 0.1,
                 exploration_initial_eps: float = 1.0,
                 exploration_final_eps: float = 0.05,
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
                                  tau, gamma, train_freq, gradient_steps,
                                  n_episodes_rollout, action_noise=None,  # No action noise
                                  policy_kwargs=policy_kwargs,
                                  tensorboard_log=tensorboard_log,
                                  verbose=verbose, device=device,
                                  create_eval_env=create_eval_env,
                                  seed=seed, sde_support=False,
                                  optimize_memory_usage=optimize_memory_usage)

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(DQN, self)._setup_model()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(self.exploration_initial_eps, self.exploration_final_eps,
                                                  self.exploration_fraction)

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self):
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollout()`` after each step in the environment.
        """
        if self.num_timesteps % self.target_update_interval == 0:
            for param, target_param in zip(self.q_net.parameters(), self.q_net_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        logger.record("rollout/exploration rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the target Q values
                target_q = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                target_q, _ = target_q.max(dim=1)
                # Avoid potential broadcast issue
                target_q = target_q.reshape(-1, 1)
                # 1-step TD target
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates
            current_q = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q = th.gather(current_q, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q, target_q)

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude='tensorboard')

    def predict(self, observation: np.ndarray,
                state: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None,
                deterministic: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: (np.ndarray) the input observation
        :param state: (Optional[np.ndarray]) The last states (can be None, used in recurrent policies)
        :param mask: (Optional[np.ndarray]) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (Tuple[np.ndarray, Optional[np.ndarray]]) the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            n_batch = observation.shape[0]
            action = np.array([self.action_space.sample() for _ in range(n_batch)])
        else:
            action, state = self.policy.predict(observation, state, mask, deterministic)
        return action, state

    def learn(self,
              total_timesteps: int,
              callback: MaybeCallback = None,
              log_interval: int = 4,
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "DQN",
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> OffPolicyAlgorithm:

        return super(DQN, self).learn(total_timesteps=total_timesteps, callback=callback, log_interval=log_interval,
                                      eval_env=eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
                                      tb_log_name=tb_log_name, eval_log_path=eval_log_path,
                                      reset_num_timesteps=reset_num_timesteps)

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
