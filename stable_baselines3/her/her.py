import functools
import inspect
from typing import Union, Type, Optional, Dict, Any, Callable, Tuple, List

import gym
import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.base_class import OffPolicyRLModel
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback

from .replay_buffer import HindsightExperienceReplayWrapper
from .utils import HERGoalEnvWrapper


def create_her(model_class: Union[Type[SAC], Type[TD3], Type[OffPolicyRLModel]] = OffPolicyRLModel):

    class HER(model_class):
        """
        Hindsight Experience Replay (HER) https://arxiv.org/abs/1707.01495
        :param policy: (BasePolicy) The policy model to use (MlpPolicy etc. )
        :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
        :param model_class: (OffPolicyRLModel) The off policy RL model to apply Hindsight Experience Replay
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
                     learning_rate: Union[float, Callable] = None,
                     **kwargs):
            if model_class == OffPolicyRLModel:
                raise Exception("Error: To initialize HER instance, the HER class needs to created with an Off-Policy \
                                 algorithm class like SAC, TD3")

            # assert issubclass(super(model_class, self), OffPolicyRLModel), \
            #     "Error: HER only works with Off policy model (such as DDPG, SAC, TD3 and DQN)."

            self.n_sampled_goal = n_sampled_goal
            self.goal_selection_strategy = goal_selection_strategy
            self.create_eval_env = kwargs.get('create_eval_env', False)
            self.model_class = model_class

            model_signature = inspect.signature(model_class.__init__)
            model_init_dict = {key: kwargs[key] for key in model_signature.parameters.keys() if key in kwargs}
            learning_rate = learning_rate or model_signature.parameters['learning_rate'].default
            # assumes all model classes have a default learning_rate

            self._create_replay_wrapper(env)
            super(HER, self).__init__(policy, self.env, learning_rate, **model_init_dict)
            self.replay_buffer = self.replay_wrapper(self.replay_buffer)
            self._save_to_file_zip = self._save_to_file

        def _create_replay_wrapper(self, env: Union[GymEnv, str]):
            """
            Wrap the environment in a HERGoalEnvWrapper
            if needed and create the replay buffer wrapper.
            """

            env = gym.make(env) if isinstance(env, str) else env

            if not isinstance(env, HERGoalEnvWrapper):
                env = HERGoalEnvWrapper(env)

            self.env = env
            self.replay_wrapper = functools.partial(HindsightExperienceReplayWrapper,
                                                    n_sampled_goal=self.n_sampled_goal,
                                                    goal_selection_strategy=self.goal_selection_strategy,
                                                    wrapped_env=self.env)
        #
        # def predict(self, observation: np.ndarray,
        #             state: Optional[np.ndarray] = None,
        #             mask: Optional[np.ndarray] = None,
        #             deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        #     return self.predict(self._check_obs(observation), state, mask, deterministic)
        #
        # def _check_obs(self, observation):
        #     if isinstance(observation, dict):
        #         if self.env is not None:
        #             if len(observation['observation'].shape) > 1:
        #                 raise NotImplementedError
        #             return self.env.convert_dict_to_obs(observation)
        #         else:
        #             raise ValueError("You must either pass an env to HER or wrap your env using HERGoalEnvWrapper")
        #     return observation

        def _save_to_file(self, save_path: str, data: Dict[str, Any] = None,
                          params: Dict[str, Any] = None, tensors: Dict[str, Any] = None) -> None:
            # HACK to save the replay wrapper
            # or better to save only the replay strategy and its params?
            # it will not work with VecEnv
            data['n_sampled_goal'] = self.n_sampled_goal
            data['goal_selection_strategy'] = self.goal_selection_strategy
            data['model_class'] = self.model_class
            data['her_obs_space'] = self.observation_space
            data['her_action_space'] = self.action_space
            data['policy'] = self.policy
            super()._save_to_file_zip(save_path, data, params)

        @classmethod
        def load(cls, load_path: str, env: Optional[GymEnv] = None, **kwargs):
            data, _, _ = cls._load_from_file(load_path)

            if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
                raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                                 "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                                  kwargs['policy_kwargs']))

            model = cls(policy=data['policy'], env=env,
                        n_sampled_goal=data['n_sampled_goal'],
                        goal_selection_strategy=data['goal_selection_strategy'],
                        _init_setup_model=True)

            model.__dict__['observation_space'] = data['her_obs_space']
            model.__dict__['action_space'] = data['her_action_space']
            model.model = data['model_class'].load(load_path, model.get_env(), **kwargs)
            model.model._save_to_file = model._save_to_file
            return model

    return HER
