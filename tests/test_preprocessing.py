import torch
from gym import spaces

from stable_baselines3.common.preprocessing import get_obs_shape, preprocess_obs


def test_get_obs_shape():
    assert get_obs_shape(spaces.Discrete(3)) == (1,)
    assert get_obs_shape(spaces.MultiDiscrete([3, 2])) == (2,)
    assert get_obs_shape(spaces.MultiBinary(3)) == (3,)
    assert get_obs_shape(spaces.MultiBinary([3, 2])) == (3, 2)
    assert get_obs_shape(spaces.Box(-2, 2, shape=(3,))) == (3,)
    assert get_obs_shape(spaces.Box(-2, 2, shape=(3, 2))) == (3, 2)


def test_preprocess_obs():
    actual = preprocess_obs(torch.tensor(2, dtype=torch.long), spaces.Discrete(3))
    expected = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)

    actual = preprocess_obs(torch.tensor([2, 0], dtype=torch.long), spaces.MultiDiscrete([3, 2]))
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)

    actual = preprocess_obs(torch.tensor([1, 0, 1], dtype=torch.long), spaces.MultiBinary(3))
    expected = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)

    actual = preprocess_obs(torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.long), spaces.MultiBinary([3, 2]))
    expected = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)

    actual = preprocess_obs(torch.tensor([1.5, 0.3, -1.8], dtype=torch.float32), spaces.Box(-2, 2, shape=(3,)))
    expected = torch.tensor([1.5, 0.3, -1.8], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)

    actual = preprocess_obs(
        torch.tensor([[1.5, 0.3, -1.8], [0.1, -0.6, -1.4]], dtype=torch.float32), spaces.Box(-2, 2, shape=(3, 2))
    )
    expected = torch.tensor([[1.5, 0.3, -1.8], [0.1, -0.6, -1.4]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)
