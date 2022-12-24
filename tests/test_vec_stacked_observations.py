import numpy as np
from gym import spaces

from stable_baselines3.common.vec_env.stacked_observations import StackedObservations

compute_stacking = StackedObservations.compute_stacking


def test_compute_stacking_box():
    space = spaces.Box(-1, 1, (4,))
    channels_first, stack_dimension, stacked_shape, repeat_axis = compute_stacking(n_stack=3, observation_space=space)
    assert not channels_first  # default is channel last
    assert stack_dimension == -1
    assert stacked_shape == (3 * 4,)
    assert repeat_axis == -1


def test_compute_stacking_multidim_box():
    space = spaces.Box(-1, 1, (4, 5))
    channels_first, stack_dimension, stacked_shape, repeat_axis = compute_stacking(n_stack=3, observation_space=space)
    assert not channels_first  # default is channel last
    assert stack_dimension == -1
    assert stacked_shape == (4, 3 * 5)
    assert repeat_axis == -1


def test_compute_stacking_multidim_box_channel_first():
    space = spaces.Box(-1, 1, (4, 5))
    channels_first, stack_dimension, stacked_shape, repeat_axis = compute_stacking(
        n_stack=3, observation_space=space, channels_order="first"
    )
    assert channels_first  # default is channel last
    assert stack_dimension == 1
    assert stacked_shape == (3 * 4, 5)
    assert repeat_axis == 0


def test_compute_stacking_image_channel_first():
    """Detect that image is channel first and stack in that dimension."""
    space = spaces.Box(0, 255, (3, 16, 24), dtype=np.uint8)
    channels_first, stack_dimension, stacked_shape, repeat_axis = compute_stacking(n_stack=3, observation_space=space)
    assert channels_first  # default is channel last
    assert stack_dimension == 1
    assert stacked_shape == (3 * 3, 16, 24)
    assert repeat_axis == 0


def test_compute_stacking_image_channel_last():
    """Detect that image is channel last and stack in that dimension."""
    space = spaces.Box(0, 255, (16, 24, 3), dtype=np.uint8)
    channels_first, stack_dimension, stacked_shape, repeat_axis = compute_stacking(n_stack=3, observation_space=space)
    assert not channels_first  # default is channel last
    assert stack_dimension == -1
    assert stacked_shape == (16, 24, 3 * 3)
    assert repeat_axis == -1


def test_compute_stacking_image_channel_first_stack_last():
    """Detect that image is channel first and stack in that dimension."""
    space = spaces.Box(0, 255, (3, 16, 24), dtype=np.uint8)
    channels_first, stack_dimension, stacked_shape, repeat_axis = compute_stacking(
        n_stack=3, observation_space=space, channels_order="last"
    )
    assert not channels_first  # default is channel last
    assert stack_dimension == -1
    assert stacked_shape == (3, 16, 3 * 24)
    assert repeat_axis == -1


def test_compute_stacking_image_channel_last_stack_first():
    """Detect that image is channel last and stack in that dimension."""
    space = spaces.Box(0, 255, (16, 24, 3), dtype=np.uint8)
    channels_first, stack_dimension, stacked_shape, repeat_axis = compute_stacking(
        n_stack=3, observation_space=space, channels_order="first"
    )
    assert channels_first  # default is channel last
    assert stack_dimension == 1
    assert stacked_shape == (3 * 16, 24, 3)
    assert repeat_axis == 0


def test_reset_update_box():
    num_envs = 2
    n_stack = 4
    space = spaces.Box(-1, 1, (4,))
    stacked_observations = StackedObservations(num_envs, n_stack, space)
    observations_1 = np.stack([space.sample() for _ in range(num_envs)])
    stacked_obs = stacked_observations.reset(observations_1)
    assert stacked_obs.shape == (num_envs, n_stack * 4)
    assert stacked_obs.dtype == space.dtype
    observations_2 = np.stack([space.sample() for _ in range(num_envs)])
    dones = np.zeros((num_envs,), dtype=bool)
    infos = [{} for _ in range(num_envs)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs.shape == (num_envs, n_stack * 4)
    assert stacked_obs.dtype == space.dtype
    assert np.array_equal(
        stacked_obs,
        np.concatenate(
            (np.zeros_like(observations_1), np.zeros_like(observations_1), observations_1, observations_2), axis=-1
        ),
    )


def test_reset_update_multidim_box():
    num_envs = 2
    n_stack = 4
    space = spaces.Box(-1, 1, (4, 5))
    stacked_observations = StackedObservations(num_envs, n_stack, space)
    observations_1 = np.stack([space.sample() for _ in range(num_envs)])
    stacked_obs = stacked_observations.reset(observations_1)
    assert stacked_obs.shape == (num_envs, 4, n_stack * 5)
    assert stacked_obs.dtype == space.dtype
    observations_2 = np.stack([space.sample() for _ in range(num_envs)])
    dones = np.zeros((num_envs,), dtype=bool)
    infos = [{} for _ in range(num_envs)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs.shape == (num_envs, 4, n_stack * 5)
    assert stacked_obs.dtype == space.dtype
    assert np.array_equal(
        stacked_obs,
        np.concatenate(
            (np.zeros_like(observations_1), np.zeros_like(observations_1), observations_1, observations_2), axis=-1
        ),
    )


def test_reset_update_multidim_box_channel_first():
    num_envs = 2
    n_stack = 4
    space = spaces.Box(-1, 1, (4, 5))
    stacked_observations = StackedObservations(num_envs, n_stack, space, channels_order="first")
    observations_1 = np.stack([space.sample() for _ in range(num_envs)])
    stacked_obs = stacked_observations.reset(observations_1)
    assert stacked_obs.shape == (num_envs, n_stack * 4, 5)
    assert stacked_obs.dtype == space.dtype
    observations_2 = np.stack([space.sample() for _ in range(num_envs)])
    dones = np.zeros((num_envs,), dtype=bool)
    infos = [{} for _ in range(num_envs)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs.shape == (num_envs, n_stack * 4, 5)
    assert stacked_obs.dtype == space.dtype
    assert np.array_equal(
        stacked_obs,
        np.concatenate((np.zeros_like(observations_1), np.zeros_like(observations_1), observations_1, observations_2), axis=1),
    )


def test_reset_update_image_channel_first():
    num_envs = 2
    n_stack = 4
    space = spaces.Box(0, 255, (3, 16, 24), dtype=np.uint8)
    stacked_observations = StackedObservations(num_envs, n_stack, space)
    observations_1 = np.stack([space.sample() for _ in range(num_envs)])
    stacked_obs = stacked_observations.reset(observations_1)
    assert stacked_obs.shape == (num_envs, n_stack * 3, 16, 24)
    assert stacked_obs.dtype == space.dtype
    observations_2 = np.stack([space.sample() for _ in range(num_envs)])
    dones = np.zeros((num_envs,), dtype=bool)
    infos = [{} for _ in range(num_envs)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs.shape == (num_envs, n_stack * 3, 16, 24)
    assert stacked_obs.dtype == space.dtype
    assert np.array_equal(
        stacked_obs,
        np.concatenate((np.zeros_like(observations_1), np.zeros_like(observations_1), observations_1, observations_2), axis=1),
    )


def test_reset_update_image_channel_last():
    num_envs = 2
    n_stack = 4
    space = spaces.Box(0, 255, (16, 24, 3), dtype=np.uint8)
    stacked_observations = StackedObservations(num_envs, n_stack, space)
    observations_1 = np.stack([space.sample() for _ in range(num_envs)])
    stacked_obs = stacked_observations.reset(observations_1)
    assert stacked_obs.shape == (num_envs, 16, 24, n_stack * 3)
    assert stacked_obs.dtype == space.dtype
    observations_2 = np.stack([space.sample() for _ in range(num_envs)])
    dones = np.zeros((num_envs,), dtype=bool)
    infos = [{} for _ in range(num_envs)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs.shape == (num_envs, 16, 24, n_stack * 3)
    assert stacked_obs.dtype == space.dtype
    assert np.array_equal(
        stacked_obs,
        np.concatenate(
            (np.zeros_like(observations_1), np.zeros_like(observations_1), observations_1, observations_2), axis=-1
        ),
    )


def test_reset_update_image_channel_first_stack_last():
    num_envs = 2
    n_stack = 4
    space = spaces.Box(0, 255, (3, 16, 24), dtype=np.uint8)
    stacked_observations = StackedObservations(num_envs, n_stack, space, channels_order="last")
    observations_1 = np.stack([space.sample() for _ in range(num_envs)])
    stacked_obs = stacked_observations.reset(observations_1)
    assert stacked_obs.shape == (num_envs, 3, 16, n_stack * 24)
    assert stacked_obs.dtype == space.dtype
    observations_2 = np.stack([space.sample() for _ in range(num_envs)])
    dones = np.zeros((num_envs,), dtype=bool)
    infos = [{} for _ in range(num_envs)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs.shape == (num_envs, 3, 16, n_stack * 24)
    assert stacked_obs.dtype == space.dtype
    assert np.array_equal(
        stacked_obs,
        np.concatenate(
            (np.zeros_like(observations_1), np.zeros_like(observations_1), observations_1, observations_2), axis=-1
        ),
    )


def test_reset_update_image_channel_last_stack_first():
    num_envs = 2
    n_stack = 4
    space = spaces.Box(0, 255, (16, 24, 3), dtype=np.uint8)
    stacked_observations = StackedObservations(num_envs, n_stack, space, channels_order="first")
    observations_1 = np.stack([space.sample() for _ in range(num_envs)])
    stacked_obs = stacked_observations.reset(observations_1)
    assert stacked_obs.shape == (num_envs, n_stack * 16, 24, 3)
    assert stacked_obs.dtype == space.dtype
    observations_2 = np.stack([space.sample() for _ in range(num_envs)])
    dones = np.zeros((num_envs,), dtype=bool)
    infos = [{} for _ in range(num_envs)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs.shape == (num_envs, n_stack * 16, 24, 3)
    assert stacked_obs.dtype == space.dtype
    assert np.array_equal(
        stacked_obs,
        np.concatenate((np.zeros_like(observations_1), np.zeros_like(observations_1), observations_1, observations_2), axis=1),
    )


def test_reset_update_dict():
    num_envs = 2
    n_stack = 4
    space = spaces.Dict({"key1": spaces.Box(0, 255, (16, 24, 3), dtype=np.uint8), "key2": spaces.Box(-1, 1, (4, 5))})
    stacked_observations = StackedObservations(num_envs, n_stack, space, channels_order={"key1": "first", "key2": "last"})
    observations_1 = {key: np.stack([subspace.sample() for _ in range(num_envs)]) for key, subspace in space.spaces.items()}
    stacked_obs = stacked_observations.reset(observations_1)
    assert isinstance(stacked_obs, dict)
    assert stacked_obs["key1"].shape == (num_envs, n_stack * 16, 24, 3)
    assert stacked_obs["key2"].shape == (num_envs, 4, n_stack * 5)
    assert stacked_obs["key1"].dtype == space["key1"].dtype
    assert stacked_obs["key2"].dtype == space["key2"].dtype
    observations_2 = {key: np.stack([subspace.sample() for _ in range(num_envs)]) for key, subspace in space.spaces.items()}
    dones = np.zeros((num_envs,), dtype=bool)
    infos = [{} for _ in range(num_envs)]
    stacked_obs, infos = stacked_observations.update(observations_2, dones, infos)
    assert stacked_obs["key1"].shape == (num_envs, n_stack * 16, 24, 3)
    assert stacked_obs["key2"].shape == (num_envs, 4, n_stack * 5)
    assert stacked_obs["key1"].dtype == space["key1"].dtype
    assert stacked_obs["key2"].dtype == space["key2"].dtype

    assert np.array_equal(
        stacked_obs["key1"],
        np.concatenate(
            (
                np.zeros_like(observations_1["key1"]),
                np.zeros_like(observations_1["key1"]),
                observations_1["key1"],
                observations_2["key1"],
            ),
            axis=1,
        ),
    )
    assert np.array_equal(
        stacked_obs["key2"],
        np.concatenate(
            (
                np.zeros_like(observations_1["key2"]),
                np.zeros_like(observations_1["key2"]),
                observations_1["key2"],
                observations_2["key2"],
            ),
            axis=-1,
        ),
    )


def test_episode_termination_box():
    num_envs = 2
    n_stack = 4
    space = spaces.Box(-1, 1, (1,))
    stacked_observations = StackedObservations(num_envs, n_stack, space)
    observations_1 = np.stack([space.sample() for _ in range(num_envs)])
    stacked_observations.reset(observations_1)
    observations_2 = np.stack([space.sample() for _ in range(num_envs)])
    dones = np.zeros((num_envs,), dtype=bool)
    infos = [{} for _ in range(num_envs)]
    stacked_observations.update(observations_2, dones, infos)
    terminal_observation = space.sample()
    infos[1]["terminal_observation"] = terminal_observation  # episode termination in env1
    dones[1] = True
    observations_3 = np.stack([space.sample() for _ in range(num_envs)])
    stacked_obs, infos = stacked_observations.update(observations_3, dones, infos)
    zeros = np.zeros_like(observations_1[0])
    true_stacked_obs_env1 = np.concatenate((zeros, observations_1[0], observations_2[0], observations_3[0]), axis=-1)
    true_stacked_obs_env2 = np.concatenate((zeros, zeros, zeros, observations_3[1]), axis=-1)
    true_stacked_obs = np.stack((true_stacked_obs_env1, true_stacked_obs_env2))
    assert np.array_equal(true_stacked_obs, stacked_obs)


def test_episode_termination_dict():
    num_envs = 2
    n_stack = 4
    space = spaces.Dict({"key1": spaces.Box(0, 255, (16, 24, 3), dtype=np.uint8), "key2": spaces.Box(-1, 1, (4, 5))})
    stacked_observations = StackedObservations(num_envs, n_stack, space, channels_order={"key1": "first", "key2": "last"})
    observations_1 = {key: np.stack([subspace.sample() for _ in range(num_envs)]) for key, subspace in space.spaces.items()}
    stacked_observations.reset(observations_1)
    observations_2 = {key: np.stack([subspace.sample() for _ in range(num_envs)]) for key, subspace in space.spaces.items()}
    dones = np.zeros((num_envs,), dtype=bool)
    infos = [{} for _ in range(num_envs)]
    stacked_observations.update(observations_2, dones, infos)
    terminal_observation = space.sample()
    infos[1]["terminal_observation"] = terminal_observation  # episode termination in env1
    dones[1] = True
    observations_3 = {key: np.stack([subspace.sample() for _ in range(num_envs)]) for key, subspace in space.spaces.items()}
    stacked_obs, infos = stacked_observations.update(observations_3, dones, infos)

    # HERE TODO: end tests here

    # zeros = np.zeros_like(observations_1[0])
    # true_stacked_obs_env1 = np.concatenate((zeros, observations_1[0], observations_2[0], observations_3[0]), axis=-1)
    # true_stacked_obs_env2 = np.concatenate((zeros, zeros, zeros, observations_3[1]), axis=-1)
    # true_stacked_obs = np.stack((true_stacked_obs_env1, true_stacked_obs_env2))
    # assert np.array_equal(true_stacked_obs, stacked_obs)
