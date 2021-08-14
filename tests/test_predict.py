import gym
import numpy as np
import pytest
import torch as th
import torch.nn as nn

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import DummyVecEnv

MODEL_LIST = [
    PPO,
    A2C,
    TD3,
    SAC,
    DQN,
]


@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_auto_wrap(model_class):
    # test auto wrapping of env into a VecEnv

    # Use different environment for DQN
    if model_class is DQN:
        env_name = "CartPole-v0"
    else:
        env_name = "Pendulum-v0"
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    model = model_class("MlpPolicy", env)
    model.learn(100, eval_env=eval_env)


@pytest.mark.parametrize("model_class", MODEL_LIST)
@pytest.mark.parametrize("env_id", ["Pendulum-v0", "CartPole-v1"])
@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_predict(model_class, env_id, device):
    if device == "cuda" and not th.cuda.is_available():
        pytest.skip("CUDA not available")

    if env_id == "CartPole-v1":
        if model_class in [SAC, TD3]:
            return
    elif model_class in [DQN]:
        return

    # Test detection of different shapes by the predict method
    model = model_class("MlpPolicy", env_id, device=device)
    # Check that the policy is on the right device
    assert get_device(device).type == model.policy.device.type

    env = gym.make(env_id)
    vec_env = DummyVecEnv([lambda: gym.make(env_id), lambda: gym.make(env_id)])

    obs = env.reset()
    action, _ = model.predict(obs)
    assert action.shape == env.action_space.shape
    assert env.action_space.contains(action)

    vec_env_obs = vec_env.reset()
    action, _ = model.predict(vec_env_obs)
    assert action.shape[0] == vec_env_obs.shape[0]

    # Special case for DQN to check the epsilon greedy exploration
    if model_class == DQN:
        model.exploration_rate = 1.0
        action, _ = model.predict(obs, deterministic=False)
        assert action.shape == env.action_space.shape
        assert env.action_space.contains(action)

        action, _ = model.predict(vec_env_obs, deterministic=False)
        assert action.shape[0] == vec_env_obs.shape[0]


class FlattenBatchNormExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input and uses batch normalization.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super(FlattenBatchNormExtractor, self).__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()
        self.batch_norm = nn.BatchNorm1d(self._features_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        result = self.flatten(observations)
        result = self.batch_norm(result)
        result = self.dropout(result)
        return result


@pytest.mark.parametrize("model_class", MODEL_LIST)
@pytest.mark.parametrize("env_id", ["Pendulum-v0", "CartPole-v1"])
def test_batch_norm_dropout(model_class, env_id):

    if env_id == "CartPole-v1":
        if model_class in [SAC, TD3]:
            return
    elif model_class in [DQN]:
        return

    model_kwargs = dict(seed=1)

    if model_class in [DQN, TD3, SAC]:
        model_kwargs["learning_starts"] = 0
    else:
        model_kwargs["n_steps"] = 64

    policy_kwargs = dict(
        features_extractor_class=FlattenBatchNormExtractor,
        net_arch=[16, 16],
    )
    model = model_class("MlpPolicy", env_id, policy_kwargs=policy_kwargs, verbose=1, **model_kwargs)

    if model_class in [SAC, TD3]:
        batch_norm = model.policy.actor.features_extractor.batch_norm
    elif model_class in [PPO, A2C]:
        batch_norm = model.policy.features_extractor.batch_norm
    else:
        # DQN
        batch_norm = model.policy.q_net.features_extractor.batch_norm

    # batch norm param before training
    bias_before_learn = batch_norm.bias.detach().cpu().numpy().copy()
    running_mean_before_learn = batch_norm.running_mean.detach().cpu().numpy().copy()
    model.learn(100)
    env = model.get_env()
    observation = env.reset()

    bias_after_learn = batch_norm.bias.detach().cpu().numpy()
    running_mean_after_learn = batch_norm.running_mean.detach().cpu().numpy().copy()

    # Run twice on the same observation to test if it is deterministic
    first_prediction, _ = model.predict(observation, deterministic=True)
    second_prediction, _ = model.predict(observation, deterministic=True)

    np.testing.assert_allclose(first_prediction, second_prediction)
    assert not np.allclose(bias_before_learn, bias_after_learn)
    assert not np.allclose(running_mean_before_learn, running_mean_after_learn)
