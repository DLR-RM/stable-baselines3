from typing import Optional, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy


class HerReplayBuffer(BaseBuffer):
    """
    Replay Buffer for online Hindsight Experience Replay (HER)

    :param env: (VecEnv) The training environment
    :param buffer_size: (int) The size of the buffer measured in transitions.
    :param goal_strategy: (GoalSelectionStrategy ) Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future', 'random']
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (Union[th.device, str]) PyTorch device
        to which the values will be converted
    :param n_envs: (int) Number of parallel environments
    :param her_ratio: (int) The ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
    """

    def __init__(
        self,
        env: VecEnv,
        buffer_size: int,
        goal_strategy: GoalSelectionStrategy,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        her_ratio: int = 2,
    ):

        super(HerReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs)

        self.env = env
        self.buffer_size = buffer_size

        # buffer with episodes
        self.buffer = []
        # TODO just for typing reason , need another solution
        self.observations = np.zeros((self.buffer_size, self.n_envs,) + self.obs_shape, dtype=observation_space.dtype)
        self.goal_strategy = goal_strategy
        self.her_ratio = 1 - (1.0 / (1 + her_ratio))

        # memory management
        # current size in episodes
        self.current_size = 0
        self.n_transitions_stored = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (ReplayBufferSamples)
        """
        return self._sample_transitions(batch_size)

    def _sample_transitions(self, batch_size: int) -> ReplayBufferSamples:
        """
        :param batch_size: (int) Number of element to sample
        :return: (ReplayBufferSamples)
        """

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, self.current_size, batch_size)
        buffer = np.array(self.buffer)
        # get episode lengths for selecting timesteps
        episode_lengths = np.array([len(ep) for ep in buffer[episode_idxs]])
        # select timesteps
        t_samples = np.array([np.random.choice(np.arange(ep_len)) for ep_len in episode_lengths])
        # get selected timesteps
        transitions = np.array([buffer[ep][trans] for ep, trans in zip(episode_idxs, t_samples)])
        # get her samples indices with her_ratio
        her_idxs = np.where(np.random.uniform(size=batch_size) < self.her_ratio)[0]
        # her samples episode lengths
        her_episode_lenghts = episode_lengths[her_idxs]

        # get new goals with goal selection strategy
        if self.goal_strategy == GoalSelectionStrategy.FINAL:
            # replay with final state of current episode
            last_transitions = buffer[episode_idxs[her_idxs], -1][:, 0]
            her_new_goals = [trans["achieved_goal"] for trans in last_transitions]
        elif self.goal_strategy == GoalSelectionStrategy.FUTURE:
            # replay with random state which comes from the same episode and was observed after current transition
            her_new_goals = []
            for idx, length in zip(her_idxs, her_episode_lenghts):
                # we have no transition after last transition of episode
                if t_samples[idx] + 1 < length:
                    index = np.random.choice(np.arange(t_samples[idx] + 1, length))
                    her_new_goals.append(buffer[episode_idxs[idx]][index][0]["achieved_goal"])
                else:
                    # delete index from her indices where we have no transition after current one
                    her_idxs = her_idxs[her_idxs != idx]
        elif self.goal_strategy == GoalSelectionStrategy.EPISODE:
            # replay with random state which comes from the same episode as current transition
            index = np.array([np.random.choice(np.arange(ep_len)) for ep_len in her_episode_lenghts])
            episode_transitions = buffer[episode_idxs[her_idxs], index][:, 0]
            her_new_goals = [trans["achieved_goal"] for trans in episode_transitions]
        elif self.goal_strategy == GoalSelectionStrategy.RANDOM:
            # replay with random state from the entire replay buffer
            ep_idx = np.random.randint(0, self.current_size, len(her_idxs))
            state_idx = [np.random.choice(np.arange(len(ep))) for ep in buffer[ep_idx]]
            random_transitions = buffer[ep_idx, state_idx][:, 0]
            her_new_goals = [trans["achieved_goal"] for trans in random_transitions]
        else:
            raise ValueError("Strategy for sampling goals not supported!")

        # assign new goals as desired_goals
        for idx, goal in enumerate(her_new_goals):
            transitions[her_idxs][:, 0][idx]["desired_goal"] = goal

        observations, actions, rewards, next_observations, dones = list(zip(*transitions))

        # compute new rewards with new goal
        achieved_goals = [new_obs["achieved_goal"] for new_obs in np.array(next_observations)[her_idxs]]
        new_rewards = np.array(rewards)
        new_rewards[her_idxs] = [
            self.env.env_method("compute_reward", ag, her_new_goals, None)
            for ag, new_goal in zip(achieved_goals, her_new_goals)
        ]

        # concatenate observation with (desired) goal
        obs = [np.concatenate([o["observation"], o["desired_goal"]], axis=1) for o in observations]
        new_obs = [np.concatenate([new_o["observation"], new_o["desired_goal"]], axis=1) for new_o in next_observations]

        data = (
            np.array(obs)[:, 0, :],
            np.array(actions, dtype=self.action_space.dtype)[:, 0, :],
            np.array(new_obs)[:, 0, :],
            np.array(dones, dtype=np.int8),
            new_rewards,
        )

        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray) -> None:
        """
        Add episode to replay buffer

        :param obs:
        :param next_obs:
        :param action:
        :param reward:
        :param done:

        :param episode: (list) Episode to store.
        """
        episode = list(zip(obs, action, reward, next_obs, done))

        episode_length = len(episode)

        # check if replay buffer has enough space for all transitions of episode
        if self.n_transitions_stored + episode_length <= self.size():
            self.buffer.append(episode)
            # update replay size
            self.current_size += 1
            self.n_transitions_stored += episode_length
        elif self.full:
            # if replay buffer is full take random stored episode and replace it
            idx = np.random.randint(0, self.current_size)

            if len(self.buffer[idx]) == episode_length:
                self.buffer[idx] = episode
            elif len(self.buffer[idx]) > episode_length:
                self.buffer[idx] = episode
                self.n_transitions_stored -= self.buffer[idx] - episode_length

        if self.n_transitions_stored == self.size():
            self.full = True
        else:
            self.full = False

    def get_current_episode_size(self):
        return self.current_size

    def get_current_size(self):
        return self.n_transitions_stored

    def get_transitions_stored(self):
        return self.n_transitions_stored

    def clear_buffer(self):
        self.buffer = []
