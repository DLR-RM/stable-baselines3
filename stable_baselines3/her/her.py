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


class HER(OffPolicyRLModel):
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
                 model_class: Union[Type[SAC], Type[TD3]],
                 n_sampled_goal: int = 4,
                 goal_selection_strategy: str = 'future',
                 learning_rate: Union[float, Callable] = None,
                 *args,
                 **kwargs):

        assert issubclass(model_class, OffPolicyRLModel), \
            "Error: HER only works with Off policy model (such as DDPG, SAC, TD3 and DQN)."

        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy
        self.create_eval_env = kwargs.get('create_eval_env', False)

        base_signature = inspect.signature(OffPolicyRLModel.__init__)
        base_init_dict = {key: kwargs[key] for key in base_signature.parameters.keys() if key in kwargs}

        model_signature = inspect.signature(model_class.__init__)
        model_init_dict = {key: kwargs[key] for key in model_signature.parameters.keys() if key in kwargs}
        learning_rate = learning_rate or model_signature.parameters['learning_rate'].default
        # assumes all model classes have a default learning_rate

        super(HER, self).__init__(policy, env, None, learning_rate,
                                  **base_init_dict)
        # Create and wrap the env if needed
        self._make_env(env)
        self._create_replay_wrapper(self.env)

        self.model_class = model_class
        self.model = self.model_class(policy, self.env, learning_rate, **model_init_dict)
        self.model._save_to_file_zip = self._save_to_file

    def learn(self,
              total_timesteps: int,
              callback: MaybeCallback = None,
              log_interval: int = 4,
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "HER",
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> OffPolicyRLModel:

        total_timesteps, callback = self.model._setup_learn(total_timesteps, eval_env, callback, eval_freq,
                                                      n_eval_episodes, eval_log_path, reset_num_timesteps,
                                                      tb_log_name)
        callback.on_training_start(locals(), globals())

        self.model.replay_buffer = self.replay_wrapper(self.model.replay_buffer)
        while self.model.num_timesteps < total_timesteps:
            rollout = self.model.collect_rollouts(self.model.env, n_episodes=self.model.n_episodes_rollout,
                                                  n_steps=self.model.train_freq, action_noise=self.model.action_noise,
                                                  callback=callback,
                                                  learning_starts=self.model.learning_starts,
                                                  replay_buffer=self.model.replay_buffer,
                                                  log_interval=log_interval)

            if rollout.continue_training is False:
                break

            self.model._update_current_progress(self.model.num_timesteps, total_timesteps)

            if self.model.num_timesteps > 0 and self.model.num_timesteps > self.model.learning_starts:
                gradient_steps = self.model.gradient_steps if self.model.gradient_steps > 0 else rollout.episode_timesteps
                self.model.train(gradient_steps, batch_size=self.model.batch_size)

        callback.on_training_end()
        return self

    def _create_replay_wrapper(self, env: GymEnv):
        """
        Wrap the environment in a HERGoalEnvWrapper
        if needed and create the replay buffer wrapper.
        """

        if not isinstance(env, HERGoalEnvWrapper):
            env = HERGoalEnvWrapper(env)

        self.env = env
        self.replay_wrapper = functools.partial(HindsightExperienceReplayWrapper,
                                                n_sampled_goal=self.n_sampled_goal,
                                                goal_selection_strategy=self.goal_selection_strategy,
                                                wrapped_env=self.env)

    def _make_env(self, env: Union[str, GymEnv]):
        # Using this to pass to pass to HER wrapper

        if env is not None:
            if isinstance(env, str):
                env = gym.make(env)

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.env = env

    def predict(self, observation: np.ndarray,
                state: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None,
                deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.model.predict(self._check_obs(observation), state, mask, deterministic)

    def _check_obs(self, observation):
        if isinstance(observation, dict):
            if self.env is not None:
                if len(observation['observation'].shape) > 1:
                    raise NotImplementedError
                return self.env.convert_dict_to_obs(observation)
            else:
                raise ValueError("You must either pass an env to HER or wrap your env using HERGoalEnvWrapper")
        return observation

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
        data['policy'] = self.model.policy
        super()._save_to_file_zip(save_path, data, params)

    def save(self, path: str, exclude: Optional[List[str]] = None, include: Optional[List[str]] = None) -> None:
        self.model.save(path, exclude, include)

    @classmethod
    def load(cls, load_path: str, env: Optional[GymEnv] = None, **kwargs):
        data, _, _ = cls._load_from_file(load_path)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data['policy'], env=env, model_class=data['model_class'],
                    n_sampled_goal=data['n_sampled_goal'],
                    goal_selection_strategy=data['goal_selection_strategy'],
                    _init_setup_model=False)
        model.__dict__['observation_space'] = data['her_obs_space']
        model.__dict__['action_space'] = data['her_action_space']
        model.model = data['model_class'].load(load_path, model.get_env(), **kwargs)
        model.model._save_to_file = model._save_to_file
        return model
