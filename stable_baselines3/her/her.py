import io
import pathlib
from typing import Callable, Iterable, List, Optional, Tuple, Type, Union

import gym
import numpy as np

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


def get_time_limit(env: VecEnv, current_max_episode_length: Optional[int]) -> int:
    """
    Get time limit from environment.

    :param env: (VecEnv) Environment from which we want to get the time limit.
    :param current_max_episode_length: (int) Current value for max_episode_length.
    :return: (int) max episode length
    """
    # try to get the attribute from environment
    if current_max_episode_length is None:
        try:
            current_max_episode_length = env.get_attr("spec")[0].max_episode_steps
        # if not available check if a valid value was passed as an argument
        except AttributeError:
            raise ValueError(
                "The max episode length could not be inferred."
                "You must specify a `max_episode_steps` when registering the environment, "
                "use a `gym.wrappers.TimeLimit` wrapper "
                "or pass `max_episode_length` to the model constructor"
            )
    return current_max_episode_length


class HER(BaseAlgorithm):
    """
    Hindsight Experience Replay (HER)

    Paper: https://arxiv.org/abs/1707.01495

    :param policy: (BasePolicy or str) The policy model to use.
    :param env: (GymEnv or str) The environment to learn from (if registered in Gym, can be str)
    :param model_class: (OffPolicyAlgorithm) Off policy model which will be used with hindsight experience replay. (SAC, TD3)
    :param n_sampled_goal: (int) Number of sampled goals for replay. (offline sampling)
    :param goal_selection_strategy: (GoalSelectionStrategy or str) Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future', 'random']
    :param online_sampling: (bool) Sample HER transitions online.
    :param learning_rate: (float or callable) learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param max_episode_length: (int) The maximum length of an episode. If not specified,
        it will be automatically inferred if the environment uses a ``gym.wrappers.TimeLimit`` wrapper
    """

    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        model_class: Type[OffPolicyAlgorithm],
        n_sampled_goal: int = 5,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        online_sampling: bool = False,
        learning_rate: Union[float, Callable] = 3e-4,
        max_episode_length: Optional[int] = None,
        *args,
        **kwargs,
    ):

        super(HER, self).__init__(policy=BasePolicy, env=env, policy_base=BasePolicy, learning_rate=learning_rate)

        # model initialization
        self.model_class = model_class
        self.model = model_class(
            policy=policy,
            env=self.env,
            learning_rate=learning_rate,
            *args,
            **kwargs,  # pytype: disable=wrong-keyword-args
        )

        self.verbose = self.model.verbose
        self.tensorboard_log = self.model.tensorboard_log

        # convert goal_selection_strategy into GoalSelectionStrategy if string
        if isinstance(goal_selection_strategy, str):
            self.goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy.lower()]
        else:
            self.goal_selection_strategy = goal_selection_strategy

        # check if goal_selection_strategy is valid
        assert isinstance(
            self.goal_selection_strategy, GoalSelectionStrategy
        ), f"Invalid goal selection strategy, please use one of {list(GoalSelectionStrategy)}"

        # maximum steps in episode
        self.max_episode_length = get_time_limit(self.env, max_episode_length)
        # storage for transitions of current episode
        self._episode_storage = HerReplayBuffer(
            self.env,
            self.max_episode_length,
            self.max_episode_length,
            self.goal_selection_strategy,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            self.n_envs,
            0.0,  # pytype: disable=wrong-arg-types
        )
        self.n_sampled_goal = n_sampled_goal

        # if we sample her transitions online use custom replay buffer
        self.online_sampling = online_sampling
        self.her_ratio = 1 - (1.0 / (self.n_sampled_goal + 1))
        # counter for steps in episode
        self.episode_steps = 0
        if self.online_sampling:
            self.model.replay_buffer = HerReplayBuffer(
                self.env,
                self.buffer_size,
                self.max_episode_length,
                self.goal_selection_strategy,
                self.env.observation_space,
                self.env.action_space,
                self.device,
                self.n_envs,
                self.her_ratio,  # pytype: disable=wrong-arg-types
            )

    def _setup_model(self) -> None:
        self.model._setup_model()

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        return self.model.predict(observation, state, mask, deterministic)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> BaseAlgorithm:

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )
        self.model.start_time = self.start_time
        self.model.ep_info_buffer = self.ep_info_buffer
        self.model.ep_success_buffer = self.ep_success_buffer
        self.model.num_timesteps = self.num_timesteps
        self.model._episode_num = self._episode_num
        self.model._last_obs = self._last_obs
        self.model._total_timesteps = self._total_timesteps

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            rollout = self.collect_rollouts(
                self.env,
                n_episodes=self.n_episodes_rollout,
                n_steps=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts and self.replay_buffer.size() > 0:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        n_episodes: int = 1,
        n_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        replay_buffer: Union[ReplayBuffer, HerReplayBuffer] = None,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ReplayBuffer.

        :param env: (VecEnv) The training environment
        :param callback: (BaseCallback) Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param n_episodes: (int) Number of episodes to use to collect rollout data
            You can also specify a ``n_steps`` instead
        :param n_steps: (int) Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param action_noise: (Optional[ActionNoise]) Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: (int) Number of steps before learning for the warm-up phase.
        :param replay_buffer: (ReplayBuffer or HerReplayBuffer)
        :param log_interval: (int) Log data every ``log_interval`` episodes
        :return: (RolloutReturn)
        """

        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

        if self.model.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:
                # concatenate observation and (desired) goal
                observation = self._last_obs
                self._last_obs = ObsDictWrapper.convert_dict(observation)

                if self.model.use_sde and self.model.sde_sample_freq > 0 and total_steps % self.model.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                self.model._last_obs = self._last_obs
                action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Perform action
                new_obs, reward, done, infos = env.step(action)

                done = done if episode_timesteps < self.max_episode_length else False

                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)
                self.model.ep_info_buffer = self.ep_info_buffer
                self.model.ep_success_buffer = self.ep_success_buffer

                # Store episode in episode storage
                if replay_buffer is not None:
                    # Store only the unnormalized version
                    if self._vec_normalize_env is not None:
                        new_obs_ = self._vec_normalize_env.get_original_obs()
                        reward_ = self._vec_normalize_env.get_original_reward()
                    else:
                        # Avoid changing the original ones
                        self._last_original_obs, new_obs_, reward_ = observation, new_obs, reward
                        self.model._last_original_obs = self._last_original_obs

                    if self.online_sampling:
                        self.replay_buffer.add(self._last_original_obs, new_obs_, buffer_action, reward_, done, infos)
                    else:
                        # add current transition to episode storage
                        self._episode_storage.add(self._last_original_obs, new_obs_, buffer_action, reward_, done, infos)

                self._last_obs = new_obs
                self.model._last_obs = self._last_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    self._last_original_obs = new_obs_
                    self.model._last_original_obs = self._last_original_obs

                self.num_timesteps += 1
                self.model.num_timesteps = self.num_timesteps
                episode_timesteps += 1
                total_steps += 1
                self.model._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self.model._on_step()

                self.episode_steps += 1

                if 0 < n_steps <= total_steps:
                    break

            if done or self.episode_steps == self.max_episode_length:
                if self.online_sampling:
                    self.replay_buffer.store_episode()
                else:
                    self._episode_storage.store_episode()
                    # store episode in replay buffer
                    self._store_transitions()
                # clear storage for current episode
                self._episode_storage.reset()

                total_episodes += 1
                self._episode_num += 1
                self.model._episode_num = self._episode_num
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

                # reset if done or episode length is reached
                self.env.reset()
                self.episode_steps = 0

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)

    def _store_transitions(self) -> None:
        """
        Store current episode in replay buffer. Sample additional goals and store new transitions in replay buffer.
        """

        # iterate over current episodes transitions
        for idx in range(self._episode_storage.size()):
            # get data of episode index
            observation = self._episode_storage.buffer["observation"][0][idx]
            desired_goal = self._episode_storage.buffer["desired_goal"][0][idx]
            next_observation = self._episode_storage.buffer["next_obs"][0][idx]
            next_achieved_goal = self._episode_storage.buffer["next_achieved_goal"][0][idx]
            next_desired_goal = self._episode_storage.buffer["next_desired_goal"][0][idx]
            action = self._episode_storage.buffer["action"][0][idx]
            reward = self._episode_storage.buffer["reward"][0][idx]
            done = self._episode_storage.buffer["done"][0][idx]
            infos = self._episode_storage.info_buffer[0][idx]

            # concatenate observation with (desired) goal
            obs = np.concatenate([observation, desired_goal], axis=-1)
            next_obs = np.concatenate([next_observation, next_desired_goal], axis=-1)
            # store data in replay buffer
            self.replay_buffer.add(obs, next_obs, action, reward, done)

            # We cannot sample a goal from the future in the last step of an episode
            if idx == self._episode_storage.size() - 1 and self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
                break

            # dimsension of observation
            obs_dim = observation.shape[1]

            for _ in range(self.n_sampled_goal):
                # sample goal
                goal = self._episode_storage.sample_goal(
                    self.goal_selection_strategy,
                    idx,
                    self._episode_storage.buffer["achieved_goal"][0],
                    self.replay_buffer.observations,
                    obs_dim,
                )

                # compute new reward with new goal
                new_reward = self.env.env_method("compute_reward", next_achieved_goal, goal, infos)

                # concatenate observation with (desired) goal
                obs = np.concatenate([observation, goal], axis=1)
                next_obs = np.concatenate([next_observation, goal], axis=1)

                # store data in replay buffer
                self.replay_buffer.add(obs, next_obs, action, new_reward, np.array([False]))

    def __getattr__(self, item):
        """
        Find attribute from model class if this class does not have it.
        """
        if hasattr(self.model, item):
            return getattr(self.model, item)
        else:
            raise AttributeError

    def get_torch_variables(self) -> Tuple[List[str], List[str]]:
        return self.model.get_torch_variables()

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: (Union[str, pathlib.Path, io.BufferedIOBase]) path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default one
        :param include: name of parameters that might be excluded but should be included anyway
        """

        # add HER parameters to model
        self.model.n_sampled_goal = self.n_sampled_goal
        self.model.goal_selection_strategy = self.goal_selection_strategy
        self.model.online_sampling = self.online_sampling
        self.model.model_class = self.model_class
        self.model.max_episode_length = self.max_episode_length

        self.model.save(path, exclude, include)

    @classmethod
    def load(cls, load_path: str, env: Optional[GymEnv] = None, **kwargs) -> "BaseAlgorithm":
        """
        Load the model from a zip-file

        :param load_path: the location of the saved data
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param kwargs: extra arguments to change the model when loading
        """
        data, params, tensors = load_from_zip_file(load_path)

        if "policy_kwargs" in data:
            for arg_to_remove in ["device"]:
                if arg_to_remove in data["policy_kwargs"]:
                    del data["policy_kwargs"][arg_to_remove]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        # check if observation space and action space are part of the saved parameters
        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")
        # check if given env is valid
        if env is not None:
            # check if wrapper for dict support is needed
            if isinstance(env.observation_space, gym.spaces.dict.Dict):
                env = ObsDictWrapper(env)
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
        # if no new env was given use stored env if possible
        if env is None and "env" in data:
            env = data["env"]

        kwargs = {}
        if "use_sde" in data and data["use_sde"]:
            kwargs["use_sde"] = True

        # noinspection PyArgumentList
        her_model = cls(
            policy=data["policy_class"],
            env=env,
            model_class=data["model_class"],
            n_sampled_goal=data["n_sampled_goal"],
            goal_selection_strategy=data["goal_selection_strategy"],
            online_sampling=data["online_sampling"],
            learning_rate=data["learning_rate"],
            max_episode_length=data["max_episode_length"],
            policy_kwargs=data["policy_kwargs"],
            _init_setup_model=True,  # pytype: disable=not-instantiable,wrong-keyword-args
            **kwargs,
        )

        # load parameters
        her_model.model.__dict__.update(data)
        her_model.__dict__.update(kwargs)

        her_model._total_timesteps = her_model.model._total_timesteps
        her_model.num_timesteps = her_model.model.num_timesteps
        her_model._episode_num = her_model.model._episode_num

        # put state_dicts back in place
        for name in params:
            attr = recursive_getattr(her_model.model, name)
            attr.load_state_dict(params[name])

        # put tensors back in place
        if tensors is not None:
            for name in tensors:
                recursive_setattr(her_model.model, name, tensors[name])

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if her_model.model.use_sde:
            her_model.model.policy.reset_noise()  # pytype: disable=attribute-error
        return her_model
