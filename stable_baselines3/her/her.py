import inspect
from typing import Union, Type, Optional, Callable, Tuple

import gym
import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import (load_from_zip_file)
from stable_baselines3.common.type_aliases import GymEnv

from .replay_buffer import HindsightExperienceReplayBuffer
from .utils import HERGoalEnvWrapper


def create_her(model_class: Union[Type[SAC], Type[TD3], Type[OffPolicyAlgorithm]] = OffPolicyAlgorithm):
    class HER(model_class):
        """
        Hindsight Experience Replay (HER) https://arxiv.org/abs/1707.01495
        :param policy: (BasePolicy) The policy model to use (MlpPolicy etc. )
        :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
        :param model_class: (OffPolicyAlgorithm) The off policy RL model to apply Hindsight Experience Replay
            currently supported: SAC, TD3
        :param n_sampled_goal: (int)
        :param goal_selection_strategy: (GoalSelectionStrategy or str)
        :param learning_rate: (float or callable) learning rate for adam optimizer,
            the same learning rate will be used for all networks (Q-Values, Actor and Value function)
            it can be a function of the current progress (from 1 to 0)
        """

        def __init__(self, policy: Type[BasePolicy],
                     env: Union[GymEnv, str],
                     n_sampled_goal: int = 4,
                     goal_selection_strategy: str = 'future',
                     add_her_while_sampling: bool = True,
                     learning_rate: Union[float, Callable] = None,
                     **kwargs):
            if model_class == OffPolicyAlgorithm:
                raise Exception("Error: To initialize HER instance, the HER class needs to created with an Off-Policy \
                                 algorithm class like SAC, TD3")

            self.n_sampled_goal = n_sampled_goal
            self.goal_selection_strategy = goal_selection_strategy
            self.create_eval_env = kwargs.get('create_eval_env', False)
            self.model_class = model_class
            self.add_her_while_sampling = add_her_while_sampling

            model_signature = inspect.signature(model_class.__init__)
            model_init_dict = {key: kwargs[key] for key in model_signature.parameters.keys() if key in kwargs}
            learning_rate = learning_rate or model_signature.parameters['learning_rate'].default
            # assumes all model classes have a default learning_rate

            self._create_her_env_wrapper(env)
            # can be removed after OffPolicyAlgorithm supports dict space

            super(HER, self).__init__(policy, self.her_wrapped_env, learning_rate, **model_init_dict)

            self.replay_buffer = HindsightExperienceReplayBuffer(self.buffer_size, self.observation_space,
                                                                 self.action_space, self.her_wrapped_env,
                                                                 self.device, 1, self.add_her_while_sampling,
                                                                 self.goal_selection_strategy, self.n_sampled_goal)

            self.her_obs_space = self.observation_space
            self.her_action_space = self.action_space

        def _create_her_env_wrapper(self, env: Union[GymEnv, str]):
            """
            Wrap the environment in a HERGoalEnvWrapper
            if needed and create the replay buffer wrapper.
            """

            env = gym.make(env) if isinstance(env, str) else env
            if not isinstance(env, HERGoalEnvWrapper):
                env = HERGoalEnvWrapper(env)

            self.her_wrapped_env = env

        def predict(self, observation: np.ndarray,
                    state: Optional[np.ndarray] = None,
                    mask: Optional[np.ndarray] = None,
                    deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            return super().predict(self._check_obs(observation), state, mask, deterministic)

        def _check_obs(self, observation):
            if isinstance(observation, dict):
                if self.env is not None:
                    if len(observation['observation'].shape) > 1:
                        raise NotImplementedError
                    return self.env.envs[0].convert_dict_to_obs(observation)
                    # Hack to handle dict. Both _check_obs and predict can be removed from HER once OffPolicyAlgorithm
                    # can handle dict obs_space
                else:
                    raise ValueError("You must either pass an env to HER or wrap your env using HERGoalEnvWrapper")
            return observation

        @classmethod
        def load(cls, load_path: str, env: Optional[GymEnv] = None):

            data, params, tensors = load_from_zip_file(load_path)
            model = cls(policy=data['policy_class'], env=env,
                        n_sampled_goal=data['n_sampled_goal'],
                        goal_selection_strategy=data['goal_selection_strategy'],
                        _init_setup_model=True)

            model.__dict__['observation_space'] = data['her_obs_space']
            model.__dict__['action_space'] = data['her_action_space']

            return model

    return HER
