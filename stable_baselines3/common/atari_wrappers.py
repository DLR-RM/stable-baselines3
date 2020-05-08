import gym
from gym.wrappers import AtariPreprocessing
import numpy as np

from stable_baselines3.common.type_aliases import GymStepReturn


class AtariWrapper(gym.Wrapper):
    """
    Atari 2600 preprocessings

    It is a wrapper around the one found in gym.
    It reshapes the observation to have an additional dimension and clip the reward.
    See https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py
    .
    This class follows the guidelines in
    Machado et al. (2018), "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents".

    Specifically:

    * NoopReset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost: turned off by default. Not recommended by Machado et al. (2018).
    * Resize to a square image: 84x84 by default
    * Grayscale observation: by default
    * Scale observation: optional

    :param env: (gym.Env) gym environment
    :param noop_max: (int): max number of no-ops
    :param frame_skip: (int): the frequency at which the agent experiences the game.
    :param screen_size: (int): resize Atari frame
    :param terminal_on_life_loss: (bool): if True, then step() returns done=True whenever a
            life is lost.
    :param grayscale_obs: (bool): if True (default), then gray scale observation is returned, otherwise, RGB observation
            is returned.
    :param scale_obs: (bool): if True, then observation normalized in range [0,1] is returned. It also limits memory
            optimization benefits of FrameStack Wrapper.
    :param scale_obs: (bool) If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    """
    def __init__(self, env: gym.Env,
                 noop_max: int = 30,
                 frame_skip: int = 4,
                 screen_size: int = 84,
                 terminal_on_life_loss: bool = False,
                 grayscale_obs: bool = True,
                 scale_obs: bool = False,
                 clip_reward: bool = True):
        env = AtariPreprocessing(env, noop_max=noop_max, frame_skip=frame_skip, screen_size=screen_size,
                                 terminal_on_life_loss=terminal_on_life_loss, grayscale_obs=grayscale_obs,
                                 scale_obs=scale_obs)
        # Add channel dimension
        if grayscale_obs:
            obs_space = env.observation_space
            _low, _high, _obs_dtype = (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
            env.observation_space = gym.spaces.Box(low=_low, high=_high, shape=obs_space.shape + (1,),
                                                   dtype=_obs_dtype)

        super(AtariWrapper, self).__init__(env)
        self.clip_reward = clip_reward

    def _add_axis(self, obs: np.ndarray) -> np.ndarray:
        if self.env.grayscale_obs:
            return obs[..., np.newaxis]
        return obs

    def reset(self) -> np.ndarray:
        return self._add_axis(self.env.reset())

    def step(self, action: int) -> GymStepReturn:
        obs, reward, done, info = self.env.step(action)
        # Bin reward to {+1, 0, -1} by its sign.
        if self.clip_reward:
            reward = np.sign(reward)
        return self._add_axis(obs), reward, done, info
