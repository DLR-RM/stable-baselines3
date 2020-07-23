from enum import Enum
from inspect import signature
from typing import Any, Callable, Dict, Optional, Type, Union

import numpy as np
import torch as th

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.her.obs_wrapper import ObsWrapper


class HER(OffPolicyAlgorithm):
    """
    Hindsight Experience Replay (HER)

    :param policy: (BasePolicy) The policy model to use.
    :param env: (VecEnv) The environment to learn from.
    :param model: (OffPolicyAlgorithm) Off policy model which will be used with hindsight experience replay. (SAC, TD3)
    :param n_goals: (int) Number of sampled goals for replay.
    :param goal_strategy: (GoalSelectionStrategy or str) Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future', 'random']
    :param online_sampling: (bool) Sample HER transitions online.
    :her_ratio: (int) The ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
    :param learning_rate: (float or callable) learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: (float) the discount factor
    :param train_freq: (int) Update the model every ``train_freq`` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param n_episodes_rollout: (int) Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``
    :param action_noise: (ActionNoise) the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param optimize_memory_usage: (bool) Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: (bool) Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param sde_support: (bool) Whether the model support gSDE or not
    """

    def __init__(
        self,
        policy: Type[BasePolicy],
        env: VecEnv,
        model: Type[OffPolicyAlgorithm],
        n_goals: int = 5,
        goal_strategy: Union[GoalSelectionStrategy, str] = "final",
        online_sampling: bool = False,
        her_ratio: int = 2,
        learning_rate: Union[float, Callable] = 3e-4,
        buffer_size: int = int(1e6),
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        n_episodes_rollout: int = -1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "cpu",
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        *args,
        **kwargs
    ):

        if isinstance(goal_strategy, str):
            self.goal_strategy = KEY_TO_GOAL_STRATEGY[goal_strategy.lower()]
        else:
            self.goal_strategy = goal_strategy

        assert isinstance(
            self.goal_strategy, GoalSelectionStrategy
        ), "Invalid goal selection strategy," "please use one of {}".format(list(GoalSelectionStrategy))

        self.env = ObsWrapper(env)

        # get arguments for the model initialization
        model_signature = signature(model.__init__)
        arguments = locals()
        model_init_dict = {
            key: arguments[key]
            for key in model_signature.parameters.keys()
            if key in arguments and key != "self" and key != "env"
        }

        super(HER, self).__init__(
            policy,
            self.env,
            BasePolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            n_episodes_rollout,
            action_noise,
            optimize_memory_usage,
            policy_kwargs,
            tensorboard_log,
            verbose,
            device,
            support_multi_env,
            create_eval_env,
            monitor_wrapper,
            seed,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            sde_support,
        )

        # model initialization
        self.model = model(env=self.env, **model_init_dict, **kwargs)

        self.online_sampling = online_sampling
        if self.online_sampling:
            self.model.replay_buffer = HerReplayBuffer(
                self.env,
                buffer_size,
                self.goal_strategy,
                self.env.observation_space,
                self.env.action_space,
                device,
                self.n_envs,
                her_ratio,
            )

        # storage for transitions of current episode
        self.episode_storage = []
        self.n_goals = n_goals

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
    ) -> "OffPolicyAlgorithm":

        total_timesteps, callback = self.model._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.model.num_timesteps < total_timesteps:

            rollout = self.collect_rollouts(
                self.env,
                n_episodes=self.model.n_episodes_rollout,
                n_steps=self.model.train_freq,
                action_noise=self.model.action_noise,
                callback=callback,
                learning_starts=self.model.learning_starts,
                replay_buffer=self.model.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.model.num_timesteps > 0 and self.model.num_timesteps > self.model.learning_starts:
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
        replay_buffer: Optional[ReplayBuffer] = None,
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
        :param replay_buffer: (ReplayBuffer)
        :param log_interval: (int) Log data every ``log_interval`` episodes
        :return: (RolloutReturn)
        """
        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

        if self.use_sde:
            self.model.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:
                # concatenate observation and (desired) goal
                observation = self.model._last_obs
                self.model._last_obs = np.concatenate([observation["observation"], observation["desired_goal"]], axis=1)

                if self.use_sde and self.sde_sample_freq > 0 and total_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.model.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self.model._sample_action(learning_starts, action_noise)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)

                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self.model._update_info_buffer(infos, done)

                # Store episode in episode storage
                if replay_buffer is not None:
                    # Store only the unnormalized version
                    if self.model._vec_normalize_env is not None:
                        new_obs_ = self.model._vec_normalize_env.get_original_obs()
                        reward_ = self.model._vec_normalize_env.get_original_reward()
                    else:
                        # Avoid changing the original ones
                        self.model._last_original_obs, new_obs_, reward_ = observation, new_obs, reward

                    # add current transition to episode storage
                    self.episode_storage.append((self.model._last_original_obs, buffer_action, reward_, new_obs_, done))

                self.model._last_obs = new_obs
                # Save the unnormalized observation
                if self.model._vec_normalize_env is not None:
                    self.model._last_original_obs = new_obs_

                self.model.num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1
                self.model._update_current_progress_remaining(self.model.num_timesteps, self.model._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self.model._on_step()

                if 0 < n_steps <= total_steps:
                    break

            if done:

                if self.online_sampling:
                    self.model.replay_buffer.add(self.episode_storage)
                else:
                    # store episode in replay buffer
                    self.store_transitions()
                # clear storage for current episode
                self.episode_storage = []

                total_episodes += 1
                self.model._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self.model._episode_num % log_interval == 0:
                    self.model._dump_logs()

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        self.model.train(gradient_steps=gradient_steps, batch_size=batch_size)

    def sample_goals(self, sample_idx: int) -> Union[np.ndarray, None]:
        """
        Sample a goal based on goal_strategy.

        :param sample_idx: (int) Index of current transition.
        :return: (np.ndarray or None) Return sampled goal.
        """
        if self.goal_strategy == GoalSelectionStrategy.FINAL:
            # replay with final state of current episode
            return self.episode_storage[-1][0]["achieved_goal"]
        elif self.goal_strategy == GoalSelectionStrategy.FUTURE:
            # replay with random state which comes from the same episode and was observed after current transition
            # we have no transition after last transition of episode

            if (sample_idx + 1) < len(self.episode_storage):
                index = np.random.choice(np.arange(sample_idx + 1, len(self.episode_storage)))
                return self.episode_storage[index][0]["achieved_goal"]
        elif self.goal_strategy == GoalSelectionStrategy.EPISODE:
            # replay with random state which comes from the same episode as current transition
            index = np.random.choice(np.arange(len(self.episode_storage)))
            return self.episode_storage[index][0]["achieved_goal"]
        elif self.goal_strategy == GoalSelectionStrategy.RANDOM:
            # replay with random state from the entire replay buffer
            index = np.random.choice(np.arange(self.model.replay_buffer.size()))
            obs = self.model.replay_buffer.observations[index]
            # get only the observation part
            obs_array = obs[:, : self.env.obs_dim]
            return obs_array
        else:
            raise ValueError("Strategy for sampling goals not supported!")

    def store_transitions(self) -> None:
        """
        Store current episode in replay buffer. Sample additional goals and store new transitions in replay buffer.
        """

        # iterate over current episodes transitions
        for idx, trans in enumerate(self.episode_storage):

            observation, action, reward, new_observation, done = trans

            # concatenate observation with (desired) goal
            obs = np.concatenate([observation["observation"], observation["desired_goal"]], axis=1)
            new_obs = np.concatenate([new_observation["observation"], new_observation["desired_goal"]], axis=1)

            # store data in replay buffer
            self.model.replay_buffer.add(obs, new_obs, action, reward, done)

            # sample set of additional goals
            sampled_goals = [sample for sample in (self.sample_goals(idx) for i in range(self.n_goals)) if sample is not None]

            # iterate over sampled goals and store new transitions in replay buffer
            for goal in sampled_goals:
                # compute new reward with new goal
                new_reward = self.env.env_method("compute_reward", new_observation["achieved_goal"], goal, None)

                # concatenate observation with (desired) goal
                obs = np.concatenate([observation["observation"], goal], axis=1)
                new_obs = np.concatenate([new_observation["observation"], goal], axis=1)

                # store data in replay buffer
                self.model.replay_buffer.add(obs, new_obs, action, new_reward, done)
