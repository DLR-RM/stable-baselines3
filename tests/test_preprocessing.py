from gym import spaces

from stable_baselines3.common.preprocessing import get_obs_shape


def test_get_obs_shape():
    assert get_obs_shape(spaces.Discrete(3)) == (1,)
    assert get_obs_shape(spaces.MultiDiscrete([3, 2])) == (2,)
    assert get_obs_shape(spaces.MultiBinary(3)) == (3,)
    assert get_obs_shape(spaces.MultiBinary([3, 2])) == (3, 2)
    assert get_obs_shape(spaces.Box(-2, 2, shape=(3,))) == (3,)
    assert get_obs_shape(spaces.Box(-2, 2, shape=(3, 2))) == (3, 2)
