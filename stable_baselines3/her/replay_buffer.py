import copy
from enum import Enum
from typing import Union, Optional

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """
    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2
    # Select a goal that was achieved
    # at some point in the training procedure
    # (and that is present in the replay buffer)
    RANDOM = 3


# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    'future': GoalSelectionStrategy.FUTURE,
    'final': GoalSelectionStrategy.FINAL,
    'episode': GoalSelectionStrategy.EPISODE,
    'random': GoalSelectionStrategy.RANDOM
}


class HindsightExperienceReplayWrapper:
    """
    Wrapper around a replay buffer in order to use HER.
    This implementation is inspired by to the one found in https://github.com/NervanaSystems/coach/.

    :param replay_buffer: (ReplayBuffer)
    :param n_sampled_goal: (int) The number of artificial transitions to generate for each actual transition
    :param goal_selection_strategy: (GoalSelectionStrategy) The method that will be used to generate
        the goals for the artificial transitions.
    :param wrapped_env: (HERGoalEnvWrapper) the GoalEnv wrapped using HERGoalEnvWrapper,
        that enables to convert observation to dict, and vice versa
    """

    def __init__(self, replay_buffer, n_sampled_goal, goal_selection_strategy, wrapped_env,
                 add_her_while_sampling=False):
        super(HindsightExperienceReplayWrapper, self).__init__()

        assert goal_selection_strategy in KEY_TO_GOAL_STRATEGY.keys(), "Invalid goal selection strategy," \
                                                                       "please use one of {}".format(
            list(KEY_TO_GOAL_STRATEGY.keys()))

        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy]
        self.env = wrapped_env
        # Buffer for storing transitions of the current episode
        self.episode_transitions = []
        self.replay_buffer = replay_buffer
        self.add_her_while_sampling = add_her_while_sampling

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (np.ndarray) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (np.ndarray) the new observation
        :param done: (bool) is the episode done
        """
        assert self.replay_buffer is not None
        # Update current episode buffer
        self.episode_transitions.append((obs_t, action, reward, obs_tp1, done))
        if done:
            # Add transitions (and imagined ones) to buffer only when an episode is over
            self._store_episode()
            # Reset episode buffer
            self.episode_transitions = []

    def sample(self, *args, **kwargs):
        if not self.add_her_while_sampling:
            return self.replay_buffer.sample(*args, **kwargs)
        else:
            samples = self.replay_buffer.sample(*args, **kwargs)

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return self.replay_buffer.can_sample(n_samples)

    def __len__(self):
        return len(self.replay_buffer)

    def _sample_achieved_goal(self, episode_transitions, transition_idx):
        """
        Sample an achieved goal according to the sampling strategy.

        :param episode_transitions: ([tuple]) a list of all the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # Sample a goal that was observed in the same episode after the current step
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # Choose the goal achieved at the end of the episode
            selected_transition = episode_transitions[-1]
        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # Random goal achieved during the episode
            selected_idx = np.random.choice(np.arange(len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.RANDOM:
            # Random goal achieved, from the entire replay buffer
            selected_idx = np.random.choice(np.arange(len(self.replay_buffer)))
            selected_transition = self.replay_buffer.storage[selected_idx]
        else:
            raise ValueError("Invalid goal selection strategy,"
                             "please use one of {}".format(list(GoalSelectionStrategy)))
        return self.env.convert_obs_to_dict(selected_transition[0][0])['achieved_goal']

    def _sample_achieved_goals(self, episode_transitions, transition_idx):
        """
        Sample a batch of achieved goals according to the sampling strategy.

        :param episode_transitions: ([tuple]) list of the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        return [
            self._sample_achieved_goal(episode_transitions, transition_idx)
            for _ in range(self.n_sampled_goal)
        ]

    def _store_episode(self):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        # For each transition in the last episode,
        # create a set of artificial transitions
        for transition_idx, transition in enumerate(self.episode_transitions):

            obs_t, obs_tp1, action, reward, done = transition

            # Add to the replay buffer
            self.replay_buffer.add(obs_t, obs_tp1, action, reward, done)

            if not self.add_her_while_sampling:
                # We cannot sample a goal from the future in the last step of an episode
                if (transition_idx == len(self.episode_transitions) - 1 and
                        self.goal_selection_strategy == GoalSelectionStrategy.FUTURE):
                    break

                # Sampled n goals per transition, where n is `n_sampled_goal`
                # this is called k in the paper
                sampled_goals = self._sample_achieved_goals(self.episode_transitions, transition_idx)
                # For each sampled goals, store a new transition
                for goal in sampled_goals:
                    # Copy transition to avoid modifying the original one
                    obs, next_obs, action, reward, done = copy.deepcopy(transition)

                    # Convert concatenated obs to dict, so we can update the goals
                    obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obs[0], next_obs[0]))

                    # Update the desired goal in the transition
                    obs_dict['desired_goal'] = goal
                    next_obs_dict['desired_goal'] = goal

                    # Update the reward according to the new desired goal

                    reward = self.env.compute_reward(next_obs_dict['achieved_goal'], goal, None)
                    # Can we use achieved_goal == desired_goal?
                    done = False

                    # Transform back to ndarrays
                    obs, next_obs = map(self.env.convert_dict_to_obs, (obs_dict, next_obs_dict))

                    # Add artificial transition to the replay buffer
                    self.replay_buffer.add(obs, next_obs, action, reward, done)


class HindsightExperienceReplayBuffer(BaseBuffer):
    """
    Replay buffer used for HER

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (th.device)
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 max_episode_len: int,
                 wrapped_env,
                 device: Union[th.device, str] = 'cpu',
                 n_envs: int = 1,
                 add_her_while_sampling: bool = False,
                 goal_selection_strategy: str = 'future',
                 n_sampled_goal: int = 4,
                 ):
        super(HindsightExperienceReplayBuffer, self).__init__(buffer_size, observation_space,
                                                              action_space, device, n_envs=n_envs)

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Assume all episodes have same lengths for now
        self.max_episode_len = max_episode_len
        self.add_her_while_sampling = add_her_while_sampling
        self.goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy]
        self.n_sampled_goal = n_sampled_goal
        self.env = wrapped_env

        if not self.add_her_while_sampling:
            batch_shape = (self.buffer_size,)
            self.observations = np.zeros((*batch_shape, self.n_envs,) + self.obs_shape, dtype=np.float32)
            self.next_observations = np.zeros((*batch_shape, self.n_envs,) + self.obs_shape, dtype=np.float32)
        else:
            batch_shape = (self.buffer_size // self.max_episode_len, self.max_episode_len)
            self.observations = np.zeros((self.buffer_size // self.max_episode_len, self.max_episode_len + 1,
                                          self.n_envs,) + self.obs_shape, dtype=np.float32)
            # No need next_observations if transitions are being stored as episodes

        self.actions = np.zeros((*batch_shape, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((*batch_shape, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((*batch_shape, self.n_envs), dtype=np.float32)
        self.pos = 0
        self.episode_transitions = []

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (np.ndarray) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (np.ndarray) the new observation
        :param done: (bool) is the episode done
        """
        # Update current episode buffer
        self.episode_transitions.append((obs_t, action, reward, obs_tp1, done))
        if done:
            # Add transitions to buffer only when an episode is over
            self._store_episode()
            # Reset episode buffer
            self.episode_transitions = []

    def _get_samples(self,
                     batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None
                     ) -> ReplayBufferSamples:
        if not self.add_her_while_sampling:
            data = (self._normalize_obs(self.observations[batch_inds, 0, :], env),
                    self.actions[batch_inds, 0, :],
                    self._normalize_obs(self.next_observations[batch_inds, 0, :], env),
                    self.dones[batch_inds],
                    self._normalize_reward(self.rewards[batch_inds], env))
            return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
        else:
            '''
            Sampling inspired by https://github.com/openai/baselines/blob/master/baselines/her/her_sampler.py
            '''

            # TODO: Implement other modes
            if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
                future_p = 1 - (1. / (1 + self.n_sampled_goal))
            elif self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
                raise NotImplementedError
            # future_t is always last timestep. Rest of code is same
            elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
                raise NotImplementedError
            # future_t is random value from 0 to last timestep, again straightforward
            elif self.goal_selection_strategy == GoalSelectionStrategy.RANDOM:
                raise NotImplementedError
            # sample second set of episode + timestep indices, use those ag's as dg's for the first set... O.o
            else:
                raise ValueError("Invalid goal selection strategy,"
                                 "please use one of {}".format(list(GoalSelectionStrategy)))

            episode_inds = batch_inds  # renaming for better clarity
            batch_size = len(episode_inds)
            timestep_inds = np.random.randint(self.max_episode_len, size=batch_size)

            # Select future time indexes proportional with probability future_p. These
            # will be used for HER replay by substituting in future goals.
            her_inds = np.where(np.random.uniform(size=batch_size) < future_p)
            future_offset = np.random.uniform(size=batch_size) * (self.max_episode_len - timestep_inds)
            future_offset = future_offset.astype(int)
            future_t = (timestep_inds + 1 + future_offset)[her_inds]

            # Replace goal with achieved goal but only for the previously-selected
            # HER transitions (as defined by her_indexes). For the other transitions,
            # keep the original goal.

            observations_dict = self.env.convert_obs_to_dict(self.observations[episode_inds])
            # next_observations_dict = self.env.convert_obs_to_dict(self.observations[episode_inds + 1])
            # TODO: update single set of obs into observations and next_observations instead of repeating

            future_ag = observations_dict['achieved_goal'][her_inds, future_t, np.newaxis][0]
            observations_dict['desired_goal'][her_inds, :] = future_ag

            rewards = self.env.compute_reward(observations_dict['achieved_goal'],
                                              observations_dict['desired_goal'], None)[:, 1:].astype(np.float32)
            # Skip reward computed at initial state
            obs = self.env.convert_dict_to_obs(observations_dict)
            data = (self._normalize_obs(obs[np.arange(obs.shape[0]), timestep_inds][:, 0], env),
                    self.actions[episode_inds][np.arange(batch_size), timestep_inds][:,  0],
                    self._normalize_obs(obs[np.arange(obs.shape[0]), timestep_inds + 1][:, 0], env),
                    self.dones[episode_inds][np.arange(batch_size), timestep_inds],
                    self._normalize_reward(rewards[np.arange(batch_size), timestep_inds], env))

            return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def sample(self,
               batch_size: int,
               env: Optional[VecNormalize] = None
               ):
        """
        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """
        full_idx = self.buffer_size//self.max_episode_len if self.add_her_while_sampling else self.buffer_size
        upper_bound = full_idx if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _store_episode(self):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        if not self.add_her_while_sampling:
            # For each transition in the last episode,
            # create a set of artificial transitions
            for transition_idx, transition in enumerate(self.episode_transitions):

                obs_t, obs_tp1, action, reward, done = transition
                self._add_transition(obs_t, obs_tp1, action, reward, done)
                self.pos += 1
                if self.pos == self.buffer_size:
                    self.full = True
                    self.pos = 0

                # We cannot sample a goal from the future in the last step of an episode
                if (transition_idx == len(self.episode_transitions) - 1 and
                        self.goal_selection_strategy == GoalSelectionStrategy.FUTURE):
                    break

                # Sampled n goals per transition, where n is `n_sampled_goal`
                # this is called k in the paper
                sampled_goals = self._sample_achieved_goals(self.episode_transitions, transition_idx)
                # For each sampled goals, store a new transition
                for goal in sampled_goals:
                    # Copy transition to avoid modifying the original one
                    obs, next_obs, action, reward, done = copy.deepcopy(transition)

                    # Convert concatenated obs to dict, so we can update the goals
                    obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obs[0], next_obs[0]))

                    # Update the desired goal in the transition
                    obs_dict['desired_goal'] = goal
                    next_obs_dict['desired_goal'] = goal

                    # Update the reward according to the new desired goal

                    reward = self.env.compute_reward(next_obs_dict['achieved_goal'], goal, None)
                    # Can we use achieved_goal == desired_goal?
                    done = False

                    # Transform back to ndarrays
                    obs, next_obs = map(self.env.convert_dict_to_obs, (obs_dict, next_obs_dict))

                    # Add artificial transition to the replay buffer
                    self._add_transition(obs_t, obs_tp1, action, reward, done)
        else:

            episode_transitions_zipped = [np.array(item) for item in list(zip(*self.episode_transitions))]
            obs_t, obs_tp1, action, reward, done = episode_transitions_zipped
            self._add_transition(obs_t, obs_tp1, action, reward, done)
            self.pos += 1  # Here self.pos signifies number of episodes stored not transitions

            if self.pos == self.buffer_size//self.max_episode_len:
                self.full = True
                self.pos = 0

    def _add_transition(self, obs, next_obs, action, reward, done):

        # Add to the replay buffer
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if not self.add_her_while_sampling:
            self.observations[self.pos] = np.array(obs).copy()
            self.next_observations[self.pos] = np.array(next_obs).copy()
        else:
            self.observations[self.pos] = np.append(np.array(obs).copy(),
                                                    np.array(next_obs[np.newaxis, -1].copy()), axis=0)

    def _sample_achieved_goals(self, episode_transitions, transition_idx):
        """
        Sample a batch of achieved goals according to the sampling strategy.

        :param episode_transitions: ([tuple]) list of the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        return [
            self._sample_achieved_goal(episode_transitions, transition_idx)
            for _ in range(self.n_sampled_goal)
        ]

    def _sample_achieved_goal(self, episode_transitions, transition_idx):
        """
        Sample an achieved goal according to the sampling strategy.

        :param episode_transitions: ([tuple]) a list of all the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # Sample a goal that was observed in the same episode after the current step
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # Choose the goal achieved at the end of the episode
            selected_transition = episode_transitions[-1]
        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # Random goal achieved during the episode
            selected_idx = np.random.choice(np.arange(len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.RANDOM:
            # Random goal achieved, from the entire replay buffer
            raise NotImplementedError  # TODO: Check 'random' strategy
            selected_idx = np.random.choice(np.arange(len(self)))
            selected_transition = self.storage[selected_idx]
        else:
            raise ValueError("Invalid goal selection strategy,"
                             "please use one of {}".format(list(GoalSelectionStrategy)))
        return self.env.convert_obs_to_dict(selected_transition[0][0])['achieved_goal']
