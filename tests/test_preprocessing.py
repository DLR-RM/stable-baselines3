import numpy as np
import pytest
import torch
from gymnasium import spaces

from stable_baselines3.common.preprocessing import (
    get_obs_shape,
    is_image_space,
    is_image_space_channels_first,
    preprocess_obs,
)


def test_get_obs_shape_discrete():
    assert get_obs_shape(spaces.Discrete(3)) == (1,)


def test_get_obs_shape_multidiscrete():
    assert get_obs_shape(spaces.MultiDiscrete([3, 2])) == (2,)


def test_get_obs_shape_multibinary():
    assert get_obs_shape(spaces.MultiBinary(3)) == (3,)


def test_get_obs_shape_multidimensional_multibinary():
    assert get_obs_shape(spaces.MultiBinary([3, 2])) == (3, 2)


def test_get_obs_shape_box():
    assert get_obs_shape(spaces.Box(-2, 2, shape=(3,))) == (3,)


def test_get_obs_shape_multidimensional_box():
    assert get_obs_shape(spaces.Box(-2, 2, shape=(3, 2))) == (3, 2)


def test_preprocess_obs_discrete():
    actual = preprocess_obs(torch.tensor([2], dtype=torch.long), spaces.Discrete(3))
    expected = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_preprocess_obs_multidiscrete():
    actual = preprocess_obs(torch.tensor([[2, 0]], dtype=torch.long), spaces.MultiDiscrete([3, 2]))
    expected = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_preprocess_obs_multibinary():
    actual = preprocess_obs(torch.tensor([[1, 0, 1]], dtype=torch.long), spaces.MultiBinary(3))
    expected = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_preprocess_obs_multidimensional_multibinary():
    actual = preprocess_obs(torch.tensor([[[1, 0], [1, 1], [0, 1]]], dtype=torch.long), spaces.MultiBinary([3, 2]))
    expected = torch.tensor([[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_preprocess_obs_box():
    actual = preprocess_obs(torch.tensor([[1.5, 0.3, -1.8]], dtype=torch.float32), spaces.Box(-2, 2, shape=(3,)))
    expected = torch.tensor([[1.5, 0.3, -1.8]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_preprocess_obs_multidimensional_box():
    actual = preprocess_obs(
        torch.tensor([[[1.5, 0.3, -1.8], [0.1, -0.6, -1.4]]], dtype=torch.float32), spaces.Box(-2, 2, shape=(3, 2))
    )
    expected = torch.tensor([[[1.5, 0.3, -1.8], [0.1, -0.6, -1.4]]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


class TestIsImageSpaceFrameStacked:
    """Tests for is_image_space with higher-dimensional (frame-stacked) observation spaces.

    Regression tests for https://github.com/DLR-RM/stable-baselines3/issues/2090
    """

    def test_4d_channel_first(self):
        """4D channel-first space (stack, C, H, W) should be detected as image."""
        space = spaces.Box(0, 255, shape=(2, 3, 64, 64), dtype=np.uint8)
        assert is_image_space(space)

    def test_4d_channel_last(self):
        """4D channel-last space (stack, H, W, C) should be detected as image."""
        space = spaces.Box(0, 255, shape=(2, 64, 64, 3), dtype=np.uint8)
        assert is_image_space(space)

    def test_4d_check_channels_channel_first(self):
        """check_channels=True should correctly identify channels in 4D channel-first."""
        space = spaces.Box(0, 255, shape=(2, 3, 64, 64), dtype=np.uint8)
        assert is_image_space(space, check_channels=True)

    def test_4d_check_channels_channel_last(self):
        """check_channels=True should correctly identify channels in 4D channel-last."""
        space = spaces.Box(0, 255, shape=(2, 64, 64, 3), dtype=np.uint8)
        assert is_image_space(space, check_channels=True)

    def test_4d_odd_channels_rejected(self):
        """4D space with invalid channel count should fail check_channels."""
        space = spaces.Box(0, 255, shape=(2, 5, 64, 64), dtype=np.uint8)
        assert is_image_space(space, check_channels=False)
        assert not is_image_space(space, check_channels=True)

    def test_4d_normalized_image(self):
        """4D normalized image (float, [0,1]) should be accepted with normalized_image=True."""
        space = spaces.Box(0.0, 1.0, shape=(2, 3, 64, 64), dtype=np.float32)
        assert is_image_space(space, normalized_image=True)

    def test_4d_wrong_dtype_rejected(self):
        """4D space with float dtype should be rejected without normalized_image."""
        space = spaces.Box(0, 255, shape=(2, 3, 64, 64), dtype=np.float32)
        assert not is_image_space(space)

    def test_4d_wrong_bounds_rejected(self):
        """4D space with incorrect bounds should be rejected."""
        space = spaces.Box(0, 10, shape=(2, 3, 64, 64), dtype=np.uint8)
        assert not is_image_space(space)

    def test_5d_space(self):
        """5D space should also be detected as image (extra batch/stack dims)."""
        space = spaces.Box(0, 255, shape=(4, 2, 3, 64, 64), dtype=np.uint8)
        assert is_image_space(space)

    def test_5d_check_channels(self):
        """5D space should correctly extract channel count from last 3 dims."""
        space = spaces.Box(0, 255, shape=(4, 2, 3, 64, 64), dtype=np.uint8)
        assert is_image_space(space, check_channels=True)

    def test_3d_still_works(self):
        """3D image spaces should continue to work (no regression)."""
        space = spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8)
        assert is_image_space(space)
        assert is_image_space(space, check_channels=True)

    def test_2d_still_rejected(self):
        """2D spaces should still be rejected."""
        space = spaces.Box(0, 255, shape=(64, 64), dtype=np.uint8)
        assert not is_image_space(space)

    def test_1d_still_rejected(self):
        """1D spaces should still be rejected."""
        space = spaces.Box(0, 255, shape=(64,), dtype=np.uint8)
        assert not is_image_space(space)


class TestIsImageSpaceChannelsFirstFrameStacked:
    """Tests for is_image_space_channels_first with higher-dimensional spaces."""

    def test_4d_channel_first(self):
        """4D channel-first (stack, C, H, W) should return True."""
        space = spaces.Box(0, 255, shape=(2, 3, 64, 64), dtype=np.uint8)
        assert is_image_space_channels_first(space)

    def test_4d_channel_last(self):
        """4D channel-last (stack, H, W, C) should return False."""
        space = spaces.Box(0, 255, shape=(2, 64, 64, 3), dtype=np.uint8)
        assert not is_image_space_channels_first(space)

    def test_4d_channel_mid_warns(self):
        """4D space with ambiguous middle channel should warn."""
        space = spaces.Box(0, 255, shape=(2, 64, 3, 64), dtype=np.uint8)
        with pytest.warns(Warning):
            assert not is_image_space_channels_first(space)

    def test_3d_still_works(self):
        """3D spaces should still work correctly (no regression)."""
        space_cf = spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8)
        assert is_image_space_channels_first(space_cf)
        space_cl = spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)
        assert not is_image_space_channels_first(space_cl)
