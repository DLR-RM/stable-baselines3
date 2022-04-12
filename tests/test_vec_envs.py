import collections
import functools
import itertools
import multiprocessing

import gym
import numpy as np
import pytest

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize

N_ENVS = 3
VEC_ENV_CLASSES = [DummyVecEnv, SubprocVecEnv]
VEC_ENV_WRAPPERS = [None, VecNormalize, VecFrameStack]


class CustomGymEnv(gym.Env):
    def __init__(self, space):
        """
        Custom gym environment for testing purposes
        """
        self.action_space = space
        self.observation_space = space
        self.current_step = 0
        self.ep_length = 4

    def reset(self):
        self.current_step = 0
        self._choose_next_state()
        return self.state

    def step(self, action):
        reward = float(np.random.rand())
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def _choose_next_state(self):
        self.state = self.observation_space.sample()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.zeros((4, 4, 3))

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            self.observation_space.seed(seed)

    @staticmethod
    def custom_method(dim_0=1, dim_1=1):
        """
        Dummy method to test call to custom method
        from VecEnv

        :param dim_0: (int)
        :param dim_1: (int)
        :return: (np.ndarray)
        """
        return np.ones((dim_0, dim_1))


@pytest.mark.parametrize("vec_env_class", VEC_ENV_CLASSES)
@pytest.mark.parametrize("vec_env_wrapper", VEC_ENV_WRAPPERS)
def test_vecenv_custom_calls(vec_env_class, vec_env_wrapper):
    """Test access to methods/attributes of vectorized environments"""

    def make_env():
        return CustomGymEnv(gym.spaces.Box(low=np.zeros(2), high=np.ones(2)))

    vec_env = vec_env_class([make_env for _ in range(N_ENVS)])

    if vec_env_wrapper is not None:
        if vec_env_wrapper == VecFrameStack:
            vec_env = vec_env_wrapper(vec_env, n_stack=2)
        else:
            vec_env = vec_env_wrapper(vec_env)

    # Test seed method
    vec_env.seed(0)
    # Test render method call
    # vec_env.render()  # we need a X server  to test the "human" mode
    vec_env.render(mode="rgb_array")
    env_method_results = vec_env.env_method("custom_method", 1, indices=None, dim_1=2)
    setattr_results = []
    # Set current_step to an arbitrary value
    for env_idx in range(N_ENVS):
        setattr_results.append(vec_env.set_attr("current_step", env_idx, indices=env_idx))
    # Retrieve the value for each environment
    getattr_results = vec_env.get_attr("current_step")

    assert len(env_method_results) == N_ENVS
    assert len(setattr_results) == N_ENVS
    assert len(getattr_results) == N_ENVS

    for env_idx in range(N_ENVS):
        assert (env_method_results[env_idx] == np.ones((1, 2))).all()
        assert setattr_results[env_idx] is None
        assert getattr_results[env_idx] == env_idx

    # Call env_method on a subset of the VecEnv
    env_method_subset = vec_env.env_method("custom_method", 1, indices=[0, 2], dim_1=3)
    assert (env_method_subset[0] == np.ones((1, 3))).all()
    assert (env_method_subset[1] == np.ones((1, 3))).all()
    assert len(env_method_subset) == 2

    # Test to change value for all the environments
    setattr_result = vec_env.set_attr("current_step", 42, indices=None)
    getattr_result = vec_env.get_attr("current_step")
    assert setattr_result is None
    assert getattr_result == [42 for _ in range(N_ENVS)]

    # Additional tests for setattr that does not affect all the environments
    vec_env.reset()
    setattr_result = vec_env.set_attr("current_step", 12, indices=[0, 1])
    getattr_result = vec_env.get_attr("current_step")
    getattr_result_subset = vec_env.get_attr("current_step", indices=[0, 1])
    assert setattr_result is None
    assert getattr_result == [12 for _ in range(2)] + [0 for _ in range(N_ENVS - 2)]
    assert getattr_result_subset == [12, 12]
    assert vec_env.get_attr("current_step", indices=[0, 2]) == [12, 0]

    vec_env.reset()
    # Change value only for first and last environment
    setattr_result = vec_env.set_attr("current_step", 12, indices=[0, -1])
    getattr_result = vec_env.get_attr("current_step")
    assert setattr_result is None
    assert getattr_result == [12] + [0 for _ in range(N_ENVS - 2)] + [12]
    assert vec_env.get_attr("current_step", indices=[-1]) == [12]

    vec_env.close()


class StepEnv(gym.Env):
    def __init__(self, max_steps):
        """Gym environment for testing that terminal observation is inserted
        correctly."""
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(np.array([0]), np.array([999]), dtype="int")
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return np.array([self.current_step], dtype="int")

    def step(self, action):
        prev_step = self.current_step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return np.array([prev_step], dtype="int"), 0.0, done, {}


@pytest.mark.parametrize("vec_env_class", VEC_ENV_CLASSES)
@pytest.mark.parametrize("vec_env_wrapper", VEC_ENV_WRAPPERS)
def test_vecenv_terminal_obs(vec_env_class, vec_env_wrapper):
    """Test that 'terminal_observation' gets added to info dict upon
    termination."""
    step_nums = [i + 5 for i in range(N_ENVS)]
    vec_env = vec_env_class([functools.partial(StepEnv, n) for n in step_nums])

    if vec_env_wrapper is not None:
        if vec_env_wrapper == VecFrameStack:
            vec_env = vec_env_wrapper(vec_env, n_stack=2)
        else:
            vec_env = vec_env_wrapper(vec_env)

    zero_acts = np.zeros((N_ENVS,), dtype="int")
    prev_obs_b = vec_env.reset()
    for step_num in range(1, max(step_nums) + 1):
        obs_b, _, done_b, info_b = vec_env.step(zero_acts)
        assert len(obs_b) == N_ENVS
        assert len(done_b) == N_ENVS
        assert len(info_b) == N_ENVS
        env_iter = zip(prev_obs_b, obs_b, done_b, info_b, step_nums)
        for prev_obs, obs, done, info, final_step_num in env_iter:
            assert done == (step_num == final_step_num)
            if not done:
                assert "terminal_observation" not in info
            else:
                terminal_obs = info["terminal_observation"]

                # do some rough ordering checks that should work for all
                # wrappers, including VecNormalize
                assert np.all(prev_obs < terminal_obs)
                assert np.all(obs < prev_obs)

                if not isinstance(vec_env, VecNormalize):
                    # more precise tests that we can't do with VecNormalize
                    # (which changes observation values)
                    assert np.all(prev_obs + 1 == terminal_obs)
                    assert np.all(obs == 0)

        prev_obs_b = obs_b

    vec_env.close()


SPACES = collections.OrderedDict(
    [
        ("discrete", gym.spaces.Discrete(2)),
        ("multidiscrete", gym.spaces.MultiDiscrete([2, 3])),
        ("multibinary", gym.spaces.MultiBinary(3)),
        ("continuous", gym.spaces.Box(low=np.zeros(2), high=np.ones(2))),
    ]
)


def check_vecenv_spaces(vec_env_class, space, obs_assert):
    """Helper method to check observation spaces in vectorized environments."""

    def make_env():
        return CustomGymEnv(space)

    vec_env = vec_env_class([make_env for _ in range(N_ENVS)])
    obs = vec_env.reset()
    obs_assert(obs)

    dones = [False] * N_ENVS
    while not any(dones):
        actions = [vec_env.action_space.sample() for _ in range(N_ENVS)]
        obs, _rews, dones, _infos = vec_env.step(actions)
        obs_assert(obs)
    vec_env.close()


def check_vecenv_obs(obs, space):
    """Helper method to check observations from multiple environments each belong to
    the appropriate observation space."""
    assert obs.shape[0] == N_ENVS
    for value in obs:
        assert space.contains(value)


@pytest.mark.parametrize("vec_env_class,space", itertools.product(VEC_ENV_CLASSES, SPACES.values()))
def test_vecenv_single_space(vec_env_class, space):
    def obs_assert(obs):
        return check_vecenv_obs(obs, space)

    check_vecenv_spaces(vec_env_class, space, obs_assert)


class _UnorderedDictSpace(gym.spaces.Dict):
    """Like DictSpace, but returns an unordered dict when sampling."""

    def sample(self):
        return dict(super().sample())


@pytest.mark.parametrize("vec_env_class", VEC_ENV_CLASSES)
def test_vecenv_dict_spaces(vec_env_class):
    """Test dictionary observation spaces with vectorized environments."""
    space = gym.spaces.Dict(SPACES)

    def obs_assert(obs):
        assert isinstance(obs, collections.OrderedDict)
        assert obs.keys() == space.spaces.keys()
        for key, values in obs.items():
            check_vecenv_obs(values, space.spaces[key])

    check_vecenv_spaces(vec_env_class, space, obs_assert)

    unordered_space = _UnorderedDictSpace(SPACES)
    # Check that vec_env_class can accept unordered dict observations (and convert to OrderedDict)
    check_vecenv_spaces(vec_env_class, unordered_space, obs_assert)


@pytest.mark.parametrize("vec_env_class", VEC_ENV_CLASSES)
def test_vecenv_tuple_spaces(vec_env_class):
    """Test tuple observation spaces with vectorized environments."""
    space = gym.spaces.Tuple(tuple(SPACES.values()))

    def obs_assert(obs):
        assert isinstance(obs, tuple)
        assert len(obs) == len(space.spaces)
        for values, inner_space in zip(obs, space.spaces):
            check_vecenv_obs(values, inner_space)

    return check_vecenv_spaces(vec_env_class, space, obs_assert)


def test_subproc_start_method():
    start_methods = [None]
    # Only test thread-safe methods. Others may deadlock tests! (gh/428)
    # Note: adding unsafe `fork` method as we are now using PyTorch
    all_methods = {"forkserver", "spawn", "fork"}
    available_methods = multiprocessing.get_all_start_methods()
    start_methods += list(all_methods.intersection(available_methods))
    space = gym.spaces.Discrete(2)

    def obs_assert(obs):
        return check_vecenv_obs(obs, space)

    for start_method in start_methods:
        vec_env_class = functools.partial(SubprocVecEnv, start_method=start_method)
        check_vecenv_spaces(vec_env_class, space, obs_assert)

    with pytest.raises(ValueError, match="cannot find context for 'illegal_method'"):
        vec_env_class = functools.partial(SubprocVecEnv, start_method="illegal_method")
        check_vecenv_spaces(vec_env_class, space, obs_assert)


class CustomWrapperA(VecNormalize):
    def __init__(self, venv):
        VecNormalize.__init__(self, venv)
        self.var_a = "a"


class CustomWrapperB(VecNormalize):
    def __init__(self, venv):
        VecNormalize.__init__(self, venv)
        self.var_b = "b"

    def func_b(self):
        return self.var_b

    def name_test(self):
        return self.__class__


class CustomWrapperBB(CustomWrapperB):
    def __init__(self, venv):
        CustomWrapperB.__init__(self, venv)
        self.var_bb = "bb"


def test_vecenv_wrapper_getattr():
    def make_env():
        return CustomGymEnv(gym.spaces.Box(low=np.zeros(2), high=np.ones(2)))

    vec_env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    wrapped = CustomWrapperA(CustomWrapperBB(vec_env))
    assert wrapped.var_a == "a"
    assert wrapped.var_b == "b"
    assert wrapped.var_bb == "bb"
    assert wrapped.func_b() == "b"
    assert wrapped.name_test() == CustomWrapperBB

    double_wrapped = CustomWrapperA(CustomWrapperB(wrapped))
    _ = double_wrapped.var_a  # should not raise as it is directly defined here
    with pytest.raises(AttributeError):  # should raise due to ambiguity
        _ = double_wrapped.var_b
    with pytest.raises(AttributeError):  # should raise as does not exist
        _ = double_wrapped.nonexistent_attribute


def test_framestack_vecenv():
    """Test that framestack environment stacks on desired axis"""

    image_space_shape = [12, 8, 3]
    zero_acts = np.zeros([N_ENVS] + image_space_shape)

    transposed_image_space_shape = image_space_shape[::-1]
    transposed_zero_acts = np.zeros([N_ENVS] + transposed_image_space_shape)

    def make_image_env():
        return CustomGymEnv(
            gym.spaces.Box(
                low=np.zeros(image_space_shape),
                high=np.ones(image_space_shape) * 255,
                dtype=np.uint8,
            )
        )

    def make_transposed_image_env():
        return CustomGymEnv(
            gym.spaces.Box(
                low=np.zeros(transposed_image_space_shape),
                high=np.ones(transposed_image_space_shape) * 255,
                dtype=np.uint8,
            )
        )

    def make_non_image_env():
        return CustomGymEnv(gym.spaces.Box(low=np.zeros((2,)), high=np.ones((2,))))

    vec_env = DummyVecEnv([make_image_env for _ in range(N_ENVS)])
    vec_env = VecFrameStack(vec_env, n_stack=2)
    obs, _, _, _ = vec_env.step(zero_acts)
    vec_env.close()

    # Should be stacked on the last dimension
    assert obs.shape[-1] == (image_space_shape[-1] * 2)

    # Try automatic stacking on first dimension now
    vec_env = DummyVecEnv([make_transposed_image_env for _ in range(N_ENVS)])
    vec_env = VecFrameStack(vec_env, n_stack=2)
    obs, _, _, _ = vec_env.step(transposed_zero_acts)
    vec_env.close()

    # Should be stacked on the first dimension (note the transposing in make_transposed_image_env)
    assert obs.shape[1] == (image_space_shape[-1] * 2)

    # Try forcing dimensions
    vec_env = DummyVecEnv([make_image_env for _ in range(N_ENVS)])
    vec_env = VecFrameStack(vec_env, n_stack=2, channels_order="last")
    obs, _, _, _ = vec_env.step(zero_acts)
    vec_env.close()

    # Should be stacked on the last dimension
    assert obs.shape[-1] == (image_space_shape[-1] * 2)

    vec_env = DummyVecEnv([make_image_env for _ in range(N_ENVS)])
    vec_env = VecFrameStack(vec_env, n_stack=2, channels_order="first")
    obs, _, _, _ = vec_env.step(zero_acts)
    vec_env.close()

    # Should be stacked on the first dimension
    assert obs.shape[1] == (image_space_shape[0] * 2)

    # Test invalid channels_order
    vec_env = DummyVecEnv([make_image_env for _ in range(N_ENVS)])
    with pytest.raises(AssertionError):
        vec_env = VecFrameStack(vec_env, n_stack=2, channels_order="not_valid")

    # Test that it works with non-image envs when no channels_order is given
    vec_env = DummyVecEnv([make_non_image_env for _ in range(N_ENVS)])
    vec_env = VecFrameStack(vec_env, n_stack=2)


def test_vec_env_is_wrapped():
    # Test is_wrapped call of subproc workers
    def make_env():
        return CustomGymEnv(gym.spaces.Box(low=np.zeros(2), high=np.ones(2)))

    def make_monitored_env():
        return Monitor(CustomGymEnv(gym.spaces.Box(low=np.zeros(2), high=np.ones(2))))

    # One with monitor, one without
    vec_env = SubprocVecEnv([make_env, make_monitored_env])

    assert vec_env.env_is_wrapped(Monitor) == [False, True]

    vec_env.close()

    # One with monitor, one without
    vec_env = DummyVecEnv([make_env, make_monitored_env])

    assert vec_env.env_is_wrapped(Monitor) == [False, True]

    vec_env = VecFrameStack(vec_env, n_stack=2)
    assert vec_env.env_is_wrapped(Monitor) == [False, True]


@pytest.mark.parametrize("vec_env_class", VEC_ENV_CLASSES)
def test_vec_seeding(vec_env_class):
    def make_env():
        return CustomGymEnv(gym.spaces.Box(low=np.zeros(2), high=np.ones(2)))

    # For SubprocVecEnv check for all starting methods
    start_methods = [None]
    if vec_env_class != DummyVecEnv:
        all_methods = {"forkserver", "spawn", "fork"}
        available_methods = multiprocessing.get_all_start_methods()
        start_methods = list(all_methods.intersection(available_methods))

    for start_method in start_methods:
        if start_method is not None:
            vec_env_class = functools.partial(SubprocVecEnv, start_method=start_method)

        n_envs = 3
        vec_env = vec_env_class([make_env] * n_envs)
        # Seed with no argument
        vec_env.seed()
        obs = vec_env.reset()
        _, rewards, _, _ = vec_env.step(np.array([vec_env.action_space.sample() for _ in range(n_envs)]))
        # Seed should be different per process
        assert not np.allclose(obs[0], obs[1])
        assert not np.allclose(rewards[0], rewards[1])
        assert not np.allclose(obs[1], obs[2])
        assert not np.allclose(rewards[1], rewards[2])

        vec_env.close()
