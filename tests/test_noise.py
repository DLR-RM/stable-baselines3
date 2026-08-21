import numpy as np
import pytest

from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
    VectorizedActionNoise,
)


def test_vectorized_noise_error_messages():
    """The errors should read as a sentence rather than as a tuple.

    Each of these raises passed a class object as a second positional argument,
    so ``str(exception)`` rendered the whole tuple — quotes and angle brackets
    included — instead of the sentence that was written.
    """
    base = OrnsteinUhlenbeckActionNoise(np.zeros(2), np.ones(2))

    with pytest.raises(ValueError) as excinfo:
        VectorizedActionNoise(None, 2)
    assert len(excinfo.value.args) == 1
    assert str(excinfo.value) == "Expected base_noise to be an instance of ActionNoise, not None"

    with pytest.raises(TypeError) as excinfo:
        VectorizedActionNoise(12, 2)
    assert len(excinfo.value.args) == 1
    assert str(excinfo.value).endswith("not int")

    vec = VectorizedActionNoise(base, 2)
    with pytest.raises(ValueError) as excinfo:
        vec.noises = [base, NormalActionNoise(np.zeros(2), np.ones(2))]
    assert len(excinfo.value.args) == 1
    assert str(excinfo.value).endswith("OrnsteinUhlenbeckActionNoise")
