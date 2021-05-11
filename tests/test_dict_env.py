import gym
import numpy as np
import pytest
from gym import spaces

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.envs import BitFlippingEnv, SimpleMultiObsEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize


class DummyDictEnv(gym.Env):
    """Custom Environment for testing purposes only"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        use_discrete_actions=False,
        channel_last=False,
        nested_dict_obs=False,
        vec_only=False,
    ):
        super().__init__()
        if use_discrete_actions:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        N_CHANNELS = 1
        HEIGHT = 64
        WIDTH = 64

        if channel_last:
            obs_shape = (HEIGHT, WIDTH, N_CHANNELS)
        else:
            obs_shape = (N_CHANNELS, HEIGHT, WIDTH)

        self.observation_space = spaces.Dict(
            {
                # Image obs
                "img": spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
                # Vector obs
                "vec": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                # Discrete obs
                "discrete": spaces.Discrete(4),
            }
        )

        # For checking consistency with normal MlpPolicy
        if vec_only:
            self.observation_space = spaces.Dict(
                {
                    # Vector obs
                    "vec": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                }
            )

        if nested_dict_obs:
            # Add dictionary observation inside observation space
            self.observation_space.spaces["nested-dict"] = spaces.Dict({"nested-dict-discrete": spaces.Discrete(4)})

    def seed(self, seed=None):
        if seed is not None:
            self.observation_space.seed(seed)

    def step(self, action):
        reward = 0.0
        done = False
        return self.observation_space.sample(), reward, done, {}

    def compute_reward(self, achieved_goal, desired_goal, info):
        return np.zeros((len(achieved_goal),))

    def reset(self):
        return self.observation_space.sample()

    def render(self, mode="human"):
        pass


@pytest.mark.parametrize("model_class", [PPO, A2C])
def test_goal_env(model_class):
    env = BitFlippingEnv(n_bits=4)
    # check that goal env works for PPO/A2C that cannot use HER replay buffer
    model = model_class("MultiInputPolicy", env, n_steps=64).learn(250)
    evaluate_policy(model, model.get_env())


@pytest.mark.parametrize("model_class", [PPO, A2C, DQN, DDPG, SAC, TD3])
def test_consistency(model_class):
    """
    Make sure that dict obs with vector only vs using flatten obs is equivalent.
    This ensures notable that the network architectures are the same.
    """
    use_discrete_actions = model_class == DQN
    dict_env = DummyDictEnv(use_discrete_actions=use_discrete_actions, vec_only=True)
    dict_env = gym.wrappers.TimeLimit(dict_env, 100)
    env = gym.wrappers.FlattenObservation(dict_env)
    dict_env.seed(10)
    obs = dict_env.reset()

    kwargs = {}
    n_steps = 256

    if model_class in {A2C, PPO}:
        kwargs = dict(
            n_steps=128,
        )
    else:
        # Avoid memory error when using replay buffer
        # Reduce the size of the features and make learning faster
        kwargs = dict(
            buffer_size=250,
            train_freq=8,
            gradient_steps=1,
        )
        if model_class == DQN:
            kwargs["learning_starts"] = 0

    dict_model = model_class("MultiInputPolicy", dict_env, gamma=0.5, seed=1, **kwargs)
    action_before_learning_1, _ = dict_model.predict(obs, deterministic=True)
    dict_model.learn(total_timesteps=n_steps)

    normal_model = model_class("MlpPolicy", env, gamma=0.5, seed=1, **kwargs)
    action_before_learning_2, _ = normal_model.predict(obs["vec"], deterministic=True)
    normal_model.learn(total_timesteps=n_steps)

    action_1, _ = dict_model.predict(obs, deterministic=True)
    action_2, _ = normal_model.predict(obs["vec"], deterministic=True)

    assert np.allclose(action_before_learning_1, action_before_learning_2)
    assert np.allclose(action_1, action_2)


@pytest.mark.parametrize("model_class", [PPO, A2C, DQN, DDPG, SAC, TD3])
@pytest.mark.parametrize("channel_last", [False, True])
def test_dict_spaces(model_class, channel_last):
    """
    Additional tests for PPO/A2C/SAC/DDPG/TD3/DQN to check observation space support
    with mixed observation.
    """
    use_discrete_actions = model_class not in [SAC, TD3, DDPG]
    env = DummyDictEnv(use_discrete_actions=use_discrete_actions, channel_last=channel_last)
    env = gym.wrappers.TimeLimit(env, 100)

    kwargs = {}
    n_steps = 256

    if model_class in {A2C, PPO}:
        kwargs = dict(
            n_steps=128,
            policy_kwargs=dict(
                net_arch=[32],
                features_extractor_kwargs=dict(cnn_output_dim=32),
            ),
        )
    else:
        # Avoid memory error when using replay buffer
        # Reduce the size of the features and make learning faster
        kwargs = dict(
            buffer_size=250,
            policy_kwargs=dict(
                net_arch=[32],
                features_extractor_kwargs=dict(cnn_output_dim=32),
            ),
            train_freq=8,
            gradient_steps=1,
        )
        if model_class == DQN:
            kwargs["learning_starts"] = 0

    model = model_class("MultiInputPolicy", env, gamma=0.5, seed=1, **kwargs)

    model.learn(total_timesteps=n_steps)

    evaluate_policy(model, env, n_eval_episodes=5, warn=False)


@pytest.mark.parametrize("model_class", [PPO, A2C])
def test_multiprocessing(model_class):
    use_discrete_actions = model_class not in [SAC, TD3, DDPG]

    def make_env():
        env = DummyDictEnv(use_discrete_actions=use_discrete_actions, channel_last=False)
        env = gym.wrappers.TimeLimit(env, 100)
        return env

    env = make_vec_env(make_env, n_envs=2, vec_env_cls=SubprocVecEnv)

    kwargs = {}
    n_steps = 256

    if model_class in {A2C, PPO}:
        kwargs = dict(
            n_steps=128,
            policy_kwargs=dict(
                net_arch=[32],
                features_extractor_kwargs=dict(cnn_output_dim=32),
            ),
        )

    model = model_class("MultiInputPolicy", env, gamma=0.5, seed=1, **kwargs)

    model.learn(total_timesteps=n_steps)


@pytest.mark.parametrize("model_class", [PPO, A2C, DQN, DDPG, SAC, TD3])
@pytest.mark.parametrize("channel_last", [False, True])
def test_dict_vec_framestack(model_class, channel_last):
    """
    Additional tests for PPO/A2C/SAC/DDPG/TD3/DQN to check observation space support
    for Dictionary spaces and VecEnvWrapper using MultiInputPolicy.
    """
    use_discrete_actions = model_class not in [SAC, TD3, DDPG]
    channels_order = {"vec": None, "img": "last" if channel_last else "first"}
    env = DummyVecEnv(
        [lambda: SimpleMultiObsEnv(random_start=True, discrete_actions=use_discrete_actions, channel_last=channel_last)]
    )

    env = VecFrameStack(env, n_stack=3, channels_order=channels_order)

    kwargs = {}
    n_steps = 256

    if model_class in {A2C, PPO}:
        kwargs = dict(
            n_steps=128,
            policy_kwargs=dict(
                net_arch=[32],
                features_extractor_kwargs=dict(cnn_output_dim=32),
            ),
        )
    else:
        # Avoid memory error when using replay buffer
        # Reduce the size of the features and make learning faster
        kwargs = dict(
            buffer_size=250,
            policy_kwargs=dict(
                net_arch=[32],
                features_extractor_kwargs=dict(cnn_output_dim=32),
            ),
            train_freq=8,
            gradient_steps=1,
        )
        if model_class == DQN:
            kwargs["learning_starts"] = 0

    model = model_class("MultiInputPolicy", env, gamma=0.5, seed=1, **kwargs)

    model.learn(total_timesteps=n_steps)

    evaluate_policy(model, env, n_eval_episodes=5, warn=False)


@pytest.mark.parametrize("model_class", [PPO, A2C, DQN, DDPG, SAC, TD3])
def test_vec_normalize(model_class):
    """
    Additional tests for PPO/A2C/SAC/DDPG/TD3/DQN to check observation space support
    for GoalEnv and VecNormalize using MultiInputPolicy.
    """
    env = DummyVecEnv([lambda: BitFlippingEnv(n_bits=4, continuous=not (model_class == DQN))])
    env = VecNormalize(env)

    kwargs = {}
    n_steps = 256

    if model_class in {A2C, PPO}:
        kwargs = dict(
            n_steps=128,
            policy_kwargs=dict(
                net_arch=[32],
            ),
        )
    else:
        # Avoid memory error when using replay buffer
        # Reduce the size of the features and make learning faster
        kwargs = dict(
            buffer_size=250,
            policy_kwargs=dict(
                net_arch=[32],
            ),
            train_freq=8,
            gradient_steps=1,
        )
        if model_class == DQN:
            kwargs["learning_starts"] = 0

    model = model_class("MultiInputPolicy", env, gamma=0.5, seed=1, **kwargs)

    model.learn(total_timesteps=n_steps)

    evaluate_policy(model, env, n_eval_episodes=5, warn=False)


def test_dict_nested():
    """
    Make sure we throw an appropiate error with nested Dict observation spaces
    """
    # Test without manual wrapping to vec-env
    env = DummyDictEnv(nested_dict_obs=True)

    with pytest.raises(NotImplementedError):
        _ = PPO("MultiInputPolicy", env, seed=1)

    # Test with manual vec-env wrapping

    with pytest.raises(NotImplementedError):
        env = DummyVecEnv([lambda: DummyDictEnv(nested_dict_obs=True)])
