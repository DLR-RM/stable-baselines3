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


class HindsightExperienceReplayBuffer(BaseBuffer):
    """
    Replay buffer used for HER

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param wrapped_env Env wrapped with HER wrapper
    :param device: (th.device)
    :param n_envs: (int) Number of parallel environments
    :param add_her_while_sampling: (bool) Whether to add HER transitions while sampling or while storing
    :param goal_selection_strategy (string)
    :param n_sampled_goal (int)
    """

    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
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
        self.add_her_while_sampling = add_her_while_sampling
        self.goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy]
        self.n_sampled_goal = n_sampled_goal
        self.env = wrapped_env

        self.observations = np.zeros((self.buffer_size, self.n_envs,) + self.obs_shape, dtype=np.float32)
        if not self.add_her_while_sampling:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs,) + self.obs_shape, dtype=np.float32)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0
        self.episode_transitions = []

    def add(self, obs: np.ndarray,
            next_obs: np.ndarray,
            action: float,
            reward: float,
            done: bool):
        """
        add a new transition to the buffer

        :param obs: (np.ndarray) the last observation
        :param next_obs: (np.ndarray) the new observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param done: (bool) is the episode done
        """
        # Update current episode buffer
        self.episode_transitions.append((obs, next_obs, action, reward, done))
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
            batch_size = len(batch_inds)
            her_inds = np.where(np.random.uniform(size=batch_size) < 1 - (1. / (1 + self.n_sampled_goal)))

            obs_dict = self.env.convert_obs_to_dict(self.observations[batch_inds])
            next_obs_dict = self.env.convert_obs_to_dict(self.observations[(batch_inds + 1) % self.buffer_size])

            if self.goal_selection_strategy == GoalSelectionStrategy.RANDOM:
                desired_goal_timestep_inds = np.random.randint(0, self.pos if not self.full else self.buffer_size,
                                                               size=len(her_inds))
            else:
                end_inds = batch_inds.copy()
                while True:
                    # Get index for end of episode (where done == True)
                    # Caution: This will be an infinite loop if there is no done = True in the whole buffer
                    # This will happen if episode length > buffer length. This needs to be asserted somewhere
                    done_check = self.dones[end_inds].reshape(-1)
                    if np.all(done_check):
                        break
                    end_inds = ((end_inds + 1 - done_check) % self.buffer_size).astype(int)

                if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
                    desired_goal_timestep_inds = end_inds[her_inds]
                elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
                    # this implementation of future includes current timestep + future steps till end of episode
                    # to avoid errors when last step of the episode is sampled
                    offset = (np.random.uniform(size=batch_size) * (end_inds - batch_inds)).astype(int)
                    desired_goal_timestep_inds = (batch_inds + 1 + offset)[her_inds]
                elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
                    begin_inds = batch_inds.copy()
                    done_check = np.zeros(batch_size)
                    buffer_end = self.pos if not self.full else self.buffer_size
                    while True:
                        # Get index for beginning of episode (where done == True)
                        begin_inds = ((begin_inds - 1 + done_check) % buffer_end).astype(int)
                        done_check = self.dones[begin_inds].reshape(-1)
                        if np.all(done_check):
                            break
                    begin_inds = (begin_inds + 1) % buffer_end
                    # new episode starts 1 step after done == True. If buffer is not full, this still works since last
                    # transition needs to have done = True

                    offset = (np.random.uniform(size=batch_size) * (end_inds - begin_inds)).astype(int)
                    desired_goal_timestep_inds = (batch_inds + 1 + offset)[her_inds]
                else:
                    raise ValueError("Invalid goal selection strategy,"
                                     "please use one of {}".format(list(GoalSelectionStrategy)))

            hindsight_goals = self.env.convert_obs_to_dict(self.observations[desired_goal_timestep_inds])
            hindsight_goals = hindsight_goals['achieved_goal']

            obs_dict['desired_goal'][her_inds] = hindsight_goals
            next_obs_dict['desired_goal'][her_inds] = hindsight_goals

            # recomputing rewards
            rewards = self.env.compute_reward(next_obs_dict['achieved_goal'],
                                              next_obs_dict['desired_goal'], None).astype(np.float32)

            obs = self.env.convert_dict_to_obs(obs_dict)
            next_obs = self.env.convert_dict_to_obs(next_obs_dict)

            data = (self._normalize_obs(obs[:, 0], env),
                    self.actions[batch_inds, 0, :],
                    self._normalize_obs(next_obs[:, 0], env),
                    self.dones[batch_inds],
                    self._normalize_reward(rewards, env))

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

        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _store_episode(self):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer. This method is called only after end of episode.
        For self.add_her_while_sampling = True, only regular transitions are stored, HER transitions are created while
        sampling
        """

        # For each transition in the last episode,
        # create a set of artificial transitions
        for transition_idx, transition in enumerate(self.episode_transitions):

            obs, next_obs, action, reward, done = transition
            self._add_transition(obs, next_obs, action, reward, done)

            if not self.add_her_while_sampling:

                # We cannot sample a goal from the future in the last step of an episode
                if (transition_idx == len(self.episode_transitions) - 1 and
                        self.goal_selection_strategy == GoalSelectionStrategy.FUTURE):
                    break

                # Sampled n goals per transition, where n is `n_sampled_goal`
                # this is called k in the paper
                sampled_goals = self._sample_achieved_goals(self.episode_transitions, transition_idx)  # CHECK
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
                    self._add_transition(obs, next_obs, action, reward, done)

    def _add_transition(self, obs: np.ndarray,
                        next_obs: np.ndarray,
                        action: np.ndarray,
                        reward: Union[float, np.ndarray],
                        done: Union[bool, np.ndarray]):

        self.observations[self.pos] = np.array(obs).copy()
        if not self.add_her_while_sampling:
            self.next_observations[self.pos] = np.array(next_obs).copy()
        else:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _sample_achieved_goals(self, episode_transitions: list,
                               transition_idx: int):
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

    def _sample_achieved_goal(self, episode_transitions: list,
                              transition_idx: int):
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
            selected_idx = np.random.choice(np.arange(self.pos if not self.full else self.buffer_size))
            obs = self.observations[selected_idx]
            return self.env.convert_obs_to_dict(obs[0])['achieved_goal']
        else:
            raise ValueError("Invalid goal selection strategy,"
                             "please use one of {}".format(list(GoalSelectionStrategy)))
        return self.env.convert_obs_to_dict(selected_transition[0][0])['achieved_goal']
