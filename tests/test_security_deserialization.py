"""
Security tests for deserialization of untrusted data remediation.

These tests verify that the ``deserialization_mode`` parameter correctly gates
unsafe deserialization across all SB3 load paths:

1. ``json_to_data`` (cloudpickle inside checkpoint .zip ``data`` JSON)
2. ``load_from_pkl`` (raw pickle, e.g. replay buffers)
3. ``VecNormalize.load`` (raw pickle)
4. ``BaseAlgorithm.load`` / ``PPO.load`` etc. (zip checkpoint entry point)
"""

import base64
import io
import json
import os
import pathlib
import pickle
import warnings
import zipfile

import numpy as np
import pytest

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import (
    json_to_data,
    load_from_pkl,
    save_to_pkl,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ---------------------------------------------------------------------------
# Helpers: craft malicious payloads that would execute code on deserialization
# ---------------------------------------------------------------------------


def _make_evil_class(sentinel_path: pathlib.Path) -> type:
    """
    Return a class whose instances, when pickled and then unpickled,
    write the sentinel file.  Uses __reduce__ with a top-level function
    (not an instance method) so that both ``pickle`` and ``cloudpickle``
    trigger the side effect on load.
    """

    def _write_sentinel() -> None:
        sentinel_path.write_text("PWNED")

    class Evil:
        def __reduce__(self):
            return (_write_sentinel, ())

    return Evil


def _evil_pickle_payload(sentinel_path: pathlib.Path) -> bytes:
    """Return a pickle payload that writes the sentinel when deserialized."""
    return pickle.dumps(_make_evil_class(sentinel_path)())


def _evil_cloudpickle_payload(sentinel_path: pathlib.Path) -> str:
    """Return a base64-encoded cloudpickle payload that writes the sentinel."""
    import cloudpickle

    return base64.b64encode(cloudpickle.dumps(_make_evil_class(sentinel_path)())).decode()


def _inject_malicious_entry(zip_path: str, sentinel_path: pathlib.Path) -> None:
    """
    Open an existing SB3 checkpoint zip, inject a cloudpickle-serialized
    malicious object into the ``data`` JSON, and write back.
    """
    with zipfile.ZipFile(zip_path) as z:
        data = json.loads(z.read("data").decode())

    payload = _evil_cloudpickle_payload(sentinel_path)
    data["_malicious_"] = {":type:": "<class 'object'>", ":serialized:": payload}

    with zipfile.ZipFile(zip_path, "r") as zin:
        with zipfile.ZipFile(zip_path + ".tmp", "w") as zout:
            for name in zin.namelist():
                content = zin.read(name)
                if name == "data":
                    content = json.dumps(data).encode()
                zout.writestr(name, content)
    os.replace(zip_path + ".tmp", zip_path)


# ---------------------------------------------------------------------------
# Test 1: json_to_data blocks malicious cloudpickle in safe mode
# ---------------------------------------------------------------------------


def test_json_to_data_safe_blocks_malicious(tmp_path):
    """json_to_data with deserialization_mode='safe' must skip :serialized: entries."""
    sentinel = tmp_path / "sentinel"
    payload = _evil_cloudpickle_payload(sentinel)
    json_str = json.dumps(
        {
            "learning_rate": 0.001,  # plain JSON, should load fine
            "_evil_": {":type:": "<class 'object'>", ":serialized:": payload},
            "verbose": 0,
        }
    )

    # Safe mode: should skip the malicious entry
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = json_to_data(json_str, deserialization_mode="safe")

    assert not sentinel.exists(), "Sentinel file created: RCE was NOT blocked!"
    assert "_evil_" not in result, "Malicious entry should have been skipped"
    assert result["learning_rate"] == 0.001, "Plain JSON entries should still load"
    assert result["verbose"] == 0
    # Check that warnings were emitted (safe-mode warning + deserialization error)
    assert len(rec) >= 1, "Expected at least one warning"


def test_json_to_data_legacy_allows_malicious(tmp_path):
    """json_to_data with deserialization_mode='legacy' must still deserialize (backward compat)."""
    sentinel = tmp_path / "sentinel"
    payload = _evil_cloudpickle_payload(sentinel)
    json_str = json.dumps(
        {
            "learning_rate": 0.001,
            "_evil_": {":type:": "<class 'object'>", ":serialized:": payload},
        }
    )

    # Legacy mode: should deserialize and execute the payload
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        _ = json_to_data(json_str, deserialization_mode="legacy")

    # In legacy mode the evil payload runs
    assert sentinel.exists(), "Legacy mode should have deserialized the payload"


# ---------------------------------------------------------------------------
# Test 2: load_from_pkl blocks in safe mode
# ---------------------------------------------------------------------------


def test_load_from_pkl_safe_blocks_evil(tmp_path):
    """load_from_pkl with deserialization_mode='safe' blocks evil globals."""
    import cloudpickle

    path = tmp_path / "test.pkl"
    sentinel = tmp_path / "sentinel"
    evil_class = _make_evil_class(sentinel)

    # cloudpickle can serialize local functions; standard pickle cannot
    with open(path, "wb") as f:
        cloudpickle.dump(evil_class(), f)

    # Safe mode: should block because __reduce__ invokes a non-allowlisted global
    with pytest.raises(pickle.UnpicklingError, match="not in the safe deserialization allowlist"):
        load_from_pkl(path, deserialization_mode="safe")

    assert not sentinel.exists(), "Sentinel file created: RCE was NOT blocked!"


def test_load_from_pkl_legacy_warns(tmp_path):
    """load_from_pkl with deserialization_mode='legacy' must emit a warning."""
    path = tmp_path / "test.pkl"
    save_to_pkl(path, {"key": "value"})

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = load_from_pkl(path, deserialization_mode="legacy")
    assert result == {"key": "value"}
    assert any("pickle deserialization" in str(w.message).lower() for w in rec)


# ---------------------------------------------------------------------------
# Test 3: VecNormalize.load blocks in safe mode
# ---------------------------------------------------------------------------


def test_vec_normalize_load_safe_works(tmp_path):
    """VecNormalize.load with deserialization_mode='safe' succeeds for trusted data."""
    import gymnasium as gym

    venv = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    vn = VecNormalize(venv)
    # Run a few steps so the stats are non-trivial
    for _ in range(20):
        vn.reset()
        vn.step([vn.action_space.sample()])
    pkl_path = tmp_path / "vecnormalize.pkl"
    vn.save(str(pkl_path))

    # Safe mode: should load successfully because VecNormalize uses only allowlisted types
    loaded = VecNormalize.load(str(pkl_path), venv, deserialization_mode="safe")
    assert loaded.obs_rms.mean.shape == vn.obs_rms.mean.shape
    # Verify the running stats were preserved
    np.testing.assert_allclose(loaded.obs_rms.mean, vn.obs_rms.mean, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 4: BaseAlgorithm.load / PPO.load with malicious checkpoint
# ---------------------------------------------------------------------------


def test_ppo_load_safe_blocks_rce(tmp_path):
    """
    PPO.load with deserialization_mode='safe' must block a malicious checkpoint
    that would normally execute arbitrary code via cloudpickle.

    In safe mode, serialized entries like observation_space, action_space,
    learning_rate, etc. are skipped. The user must provide them via
    custom_objects or pass an env. Here we pass custom_objects to supply
    the required objects.
    """
    import gymnasium as gym

    sentinel = tmp_path / "sentinel"

    # Create a clean model and save
    model = PPO("MlpPolicy", "CartPole-v1", n_steps=64, device="cpu")
    zip_path = str(tmp_path / "model.zip")
    model.save(zip_path)

    # Inject malicious entry
    _inject_malicious_entry(zip_path, sentinel)

    # Safe mode: must provide custom_objects for serialized entries
    env = gym.make("CartPole-v1")
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "policy_class": PPO.policy_aliases["MlpPolicy"],
        "rollout_buffer_class": model.rollout_buffer_class,
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        loaded = PPO.load(
            zip_path,
            env=env,
            device="cpu",
            deserialization_mode="safe",
            custom_objects=custom_objects,
        )

    assert not sentinel.exists(), "Sentinel file created after PPO.load(safe): RCE was NOT blocked!"
    # Verify the loaded model works
    obs = loaded.get_env().reset()
    loaded.predict(obs, deterministic=True)


def test_ppo_load_legacy_allows_rce(tmp_path):
    """
    PPO.load with deserialization_mode='legacy' still deserializes malicious
    checkpoints (backward compatibility).
    """
    sentinel = tmp_path / "sentinel"

    model = PPO("MlpPolicy", "CartPole-v1", n_steps=64, device="cpu")
    zip_path = str(tmp_path / "model.zip")
    model.save(zip_path)

    _inject_malicious_entry(zip_path, sentinel)

    # Legacy mode: payload executes
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        PPO.load(zip_path, device="cpu", deserialization_mode="legacy")

    assert sentinel.exists(), "Legacy mode should have executed the payload"


# ---------------------------------------------------------------------------
# Test 5: Invalid deserialization_mode raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func_and_args",
    [
        # json_to_data
        lambda: (json_to_data, (json.dumps({}), None, "bad_mode")),
        # load_from_pkl
        lambda: (load_from_pkl, (io.BytesIO(pickle.dumps("x")), 0, "bad_mode")),
    ],
)
def test_invalid_deserialization_mode_raises(func_and_args):
    func, args = func_and_args()
    with pytest.raises(ValueError, match="deserialization_mode must be"):
        func(*args)


# ---------------------------------------------------------------------------
# Test 6: custom_objects overrides still work in safe mode
# ---------------------------------------------------------------------------


def test_load_replay_buffer_safe_works(tmp_path):
    """load_replay_buffer with deserialization_mode='safe' succeeds for trusted buffers."""
    from stable_baselines3 import SAC

    model = SAC("MlpPolicy", "Pendulum-v1", buffer_size=1000, learning_starts=50, device="cpu")
    model.learn(100)
    original_size = model.replay_buffer.size()
    pkl_path = str(tmp_path / "replay_buffer.pkl")
    model.save_replay_buffer(pkl_path)

    # Safe mode: should load successfully
    model.load_replay_buffer(pkl_path, deserialization_mode="safe")

    assert model.replay_buffer.size() == original_size


def test_model_load_safe_with_action_noise(tmp_path):
    """Model checkpoints with action noise load in safe mode."""
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise

    noise = NormalActionNoise(mean=[0], sigma=[0.1])
    model = DDPG(
        "MlpPolicy",
        "Pendulum-v1",
        buffer_size=1000,
        learning_starts=50,
        action_noise=noise,
        device="cpu",
    )
    model.learn(100)
    zip_path = str(tmp_path / "model_noise.zip")
    model.save(zip_path)

    # Safe mode: should load successfully with action noise intact
    loaded = DDPG.load(zip_path, device="cpu", deserialization_mode="safe")
    assert loaded.action_noise is not None
    assert isinstance(loaded.action_noise, NormalActionNoise)


def test_vec_normalize_load_safe_blocks_evil(tmp_path):
    """VecNormalize.load with deserialization_mode='safe' blocks evil globals."""
    import cloudpickle
    import gymnasium as gym

    sentinel = tmp_path / "sentinel"
    venv = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    evil_class = _make_evil_class(sentinel)
    evil_data = cloudpickle.dumps(evil_class())

    pkl_path = tmp_path / "vecnormalize_evil.pkl"
    with open(pkl_path, "wb") as f:
        f.write(evil_data)

    # Safe mode: should block the evil global
    with pytest.raises(pickle.UnpicklingError, match="not in the safe deserialization allowlist"):
        VecNormalize.load(str(pkl_path), venv, deserialization_mode="safe")

    assert not sentinel.exists(), "Sentinel file created: RCE was NOT blocked!"


def test_json_to_data_safe_with_custom_objects(tmp_path):
    """In safe mode, keys provided via custom_objects are used instead of skipping."""
    sentinel = tmp_path / "sentinel"
    payload = _evil_cloudpickle_payload(sentinel)
    json_str = json.dumps(
        {
            "learning_rate": 0.001,
            "lr_schedule": {":type:": "<class 'function'>", ":serialized:": payload},
        }
    )

    # Provide a safe replacement via custom_objects
    custom = {"lr_schedule": lambda x: 0.0}
    result = json_to_data(json_str, custom_objects=custom, deserialization_mode="safe")

    assert not sentinel.exists(), "Sentinel created: custom_objects override failed"
    assert callable(result["lr_schedule"])
    assert result["lr_schedule"](0.5) == 0.0


# ---------------------------------------------------------------------------
# Test 7: add_safe_globals and safe_globals context manager
# ---------------------------------------------------------------------------


def test_add_safe_globals_persists():
    """add_safe_globals() should persist across calls."""
    from stable_baselines3.common.safe_globals import (
        _USER_SAFE_GLOBALS,
        add_safe_globals,
        get_safe_globals,
    )

    # Clear user globals to start fresh
    _USER_SAFE_GLOBALS.clear()

    class MyCustomType:
        pass

    # Before registration: MyCustomType should NOT be in the allowlist
    before = get_safe_globals()
    qualname = f"{MyCustomType.__module__}.{MyCustomType.__qualname__}"
    assert qualname not in before, "Custom type should not be in allowlist yet"

    # Register it
    add_safe_globals([MyCustomType])

    # After registration: MyCustomType should be in the allowlist
    after = get_safe_globals()
    assert qualname in after, "Custom type should be in allowlist after add_safe_globals"

    # Clean up
    _USER_SAFE_GLOBALS.discard(qualname)


def test_safe_globals_context_manager(tmp_path):
    """safe_globals context manager should restore allowlist on exit."""
    import cloudpickle

    from stable_baselines3.common.safe_globals import (
        _USER_SAFE_GLOBALS,
        get_safe_globals,
        safe_globals,
    )

    # Clear user globals to start fresh
    _USER_SAFE_GLOBALS.clear()

    class ScopedType:
        value = 42

    qualname = f"{ScopedType.__module__}.{ScopedType.__qualname__}"

    # Before context: not in allowlist
    assert qualname not in get_safe_globals()

    # Inside context: should be in allowlist
    with safe_globals([ScopedType]):
        assert qualname in get_safe_globals()
        # Verify the type can actually be deserialized
        payload = cloudpickle.dumps(ScopedType())
        from stable_baselines3.common.save_util import _cloudpickle_loads_safe

        obj = _cloudpickle_loads_safe(payload)
        assert obj.value == 42

    # After context: should be removed
    assert qualname not in get_safe_globals(), "safe_globals context manager did not restore allowlist on exit"
