"""
Central safe-globals registration for safe deserialization (CWE-502 mitigation).

This module collects the allowlist of types that are permitted during
restricted deserialization.  The same allowlist is used by two mechanisms:

1. ``_RestrictedUnpickler`` (pickle / cloudpickle data stored inside zip
   checkpoints and in raw .pkl files).
2. ``torch.serialization.add_safe_globals`` so that ``torch.load(...,
   weights_only=True)`` accepts the same types when loading standalone
   policy pickles (.pkl) and PyTorch state-dicts (.pth) inside zip
   checkpoints.

Both mechanisms share this single source of truth so that adding a new
type to the allowlist is a one-line change.
"""

from __future__ import annotations

import collections
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import torch as th

# ---------------------------------------------------------------------------
# Lazily-populated sets (populated by _register_safe_globals)
# ---------------------------------------------------------------------------

# String allowlist for _RestrictedUnpickler (module.ClassPath).
_SAFE_GLOBALS_STR: set[str] | None = None

# Actual objects registered with torch.serialization.add_safe_globals.
# We register them at most once.
_TORCH_REGISTRATION_DONE = False


def _collect_numpy_types() -> list[type | Callable[..., Any]]:
    """Collect numpy types needed for gymnasium spaces and array reconstruction.

    For the **string allowlist** (restricted unpickler), numpy-internal pickle
    helpers are handled by ``_NUMPY_PICKLE_INTERNALS_STR`` so that both
    numpy 1.x and 2.x module paths are accepted without crashing on import.

    For **torch.serialization.add_safe_globals** (``weights_only=True``), we
    also need the *live objects* of the currently installed numpy version.
    Those are imported here inside try/except blocks so the code works on
    either numpy 1.x or 2.x.
    """
    types: list[type | Callable[..., Any]] = [
        np.ndarray,
        np.dtype,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.int8,
        np.int16,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.bool_,
    ]

    # numpy dtype descriptor classes (needed by torch weights_only)
    types += [
        np.dtypes.Float32DType,
        np.dtypes.Float64DType,
        np.dtypes.Int32DType,
        np.dtypes.Int64DType,
        np.dtypes.Int8DType,
        np.dtypes.Int16DType,
        np.dtypes.UInt8DType,
        np.dtypes.UInt16DType,
        np.dtypes.UInt32DType,
        np.dtypes.UInt64DType,
        np.dtypes.BoolDType,
    ]

    # numpy random internals (gymnasium spaces store internal RNG state)
    types += [
        np.random.bit_generator.BitGenerator,
        np.random.bit_generator.SeedSequence,
        np.random._pcg64.PCG64,
        np.random._mt19937.MT19937,
        np.random._philox.Philox,
        np.random._sfc64.SFC64,
        np.random._generator.Generator,
        np.random.mtrand.RandomState,
    ]

    # numpy pickle reconstruction helpers (functions) — try the numpy 2.x path
    # first, then fall back to the numpy 1.x path.  If neither works, we skip
    # registration with torch (the string allowlist still covers pickle).
    try:
        import numpy._core.multiarray as np_ma
        import numpy._core.numeric as np_numeric

        types += [
            np_ma._reconstruct,  # type: ignore[attr-defined]
            np_ma.scalar,  # type: ignore[attr-defined]
            np_numeric._frombuffer,  # type: ignore[attr-defined]
        ]
    except ModuleNotFoundError:  # pragma: no cover
        try:
            import numpy.core.multiarray as np_ma_np1
            import numpy.core.numeric as np_numeric_np1

            types += [
                np_ma_np1._reconstruct,  # type: ignore[attr-defined]
                np_ma_np1.scalar,  # type: ignore[attr-defined]
                np_numeric_np1._frombuffer,  # type: ignore[attr-defined]
            ]
        except ModuleNotFoundError:
            pass

    try:
        import numpy.random._pickle as np_rp

        types += [
            np_rp.__generator_ctor,
            np_rp.__bit_generator_ctor,
            np_rp.__randomstate_ctor,
        ]
    except ModuleNotFoundError:
        pass

    # Cython unpickle helpers in numpy random
    try:
        import numpy.random.bit_generator as np_bg

        types += [
            np_bg.__pyx_unpickle_SeedSequence,  # type: ignore[attr-defined]
            np_bg.__pyx_unpickle_SeedlessSeedSequence,  # type: ignore[attr-defined]
        ]
    except ModuleNotFoundError:
        pass

    return types


# ---------------------------------------------------------------------------
# Explicit string allowlist for numpy internal pickle helpers.
# These cover BOTH numpy 1.x (numpy.core.*) and numpy 2.x (numpy._core.*)
# module paths, because we cannot import them reliably at module load time.
# The restricted unpickler relies on these strings, so checkpoints saved
# with either numpy version load correctly.
# ---------------------------------------------------------------------------

_NUMPY_PICKLE_INTERNALS_STR: tuple[str, ...] = (
    # numpy 2.x paths
    "numpy._core.multiarray._reconstruct",
    "numpy._core.multiarray.scalar",
    "numpy._core.numeric._frombuffer",
    # numpy 1.x paths
    "numpy.core.multiarray._reconstruct",
    "numpy.core.multiarray.scalar",
    "numpy.core.numeric._frombuffer",
    # numpy random pickle helpers (shared across versions)
    "numpy.random._pickle.__generator_ctor",
    "numpy.random._pickle.__bit_generator_ctor",
    "numpy.random._pickle.__randomstate_ctor",
    # Cython unpickle helpers in numpy.random.bit_generator
    "numpy.random.bit_generator.__pyx_unpickle_SeedSequence",
    "numpy.random.bit_generator.__pyx_unpickle_SeedlessSeedSequence",
)


def _collect_cloudpickle_types() -> list:
    """Collect cloudpickle 2.x / 3.x internal helpers needed by torch weights_only."""
    import cloudpickle.cloudpickle as cp

    types = [
        cp._make_function,
        cp._make_cell,
        cp._make_empty_cell,
        cp._make_skeleton_class,
        cp._make_skeleton_enum,
        cp._builtin_type,
        cp._function_setstate,
    ]

    # cloudpickle 3.x extras (may not exist in older versions)
    for name in (
        "_make_dict_items",
        "_make_dict_keys",
        "_make_dict_values",
        "_make_typevar",
    ):
        if hasattr(cp, name):
            types.append(getattr(cp, name))

    return types


def _collect_base_types() -> list[type | Callable[..., Any]]:
    """Collect types that don't depend on SB3 internal imports (no circular import risk)."""
    from gymnasium import spaces
    from torch import nn

    types: list[type | Callable[..., Any]] = []

    # Gymnasium spaces
    types += [
        spaces.Box,
        spaces.Discrete,
        spaces.MultiBinary,
        spaces.MultiDiscrete,
        spaces.Dict,
        spaces.Tuple,
        spaces.Space,
    ]

    # Try to also register old gym spaces (for models saved with gym <= 0.26)
    try:  # pragma: no cover
        import gym

        types += [
            gym.spaces.Box,
            gym.spaces.Discrete,
            gym.spaces.MultiBinary,
            gym.spaces.MultiDiscrete,
            gym.spaces.Dict,
            gym.spaces.Tuple,
            gym.spaces.Space,
        ]
    except ImportError:
        pass

    # Numpy types
    types += _collect_numpy_types()

    # PyTorch types
    types += [
        th.optim.Adam,
        th.optim.SGD,
        th.optim.RMSprop,
        th.device,
    ]

    # PyTorch nn.Module subclasses used as policy constructor parameters
    types += [
        nn.Linear,
        nn.Conv2d,
        nn.Flatten,
        nn.Sequential,
        nn.Tanh,
        nn.ReLU,
        nn.ELU,
        nn.LeakyReLU,
        nn.SiLU,
        nn.GELU,
        nn.LayerNorm,
        nn.BatchNorm2d,
    ]

    # Cloudpickle internals
    types += _collect_cloudpickle_types()

    # Standard library containers and builtins
    types += [
        collections.deque,
        collections.OrderedDict,
        getattr,
        setattr,
    ]

    return types


def _get_safe_globals_str() -> set[str]:
    """Return the string allowlist for _RestrictedUnpickler, populating it lazily."""
    global _SAFE_GLOBALS_STR
    if _SAFE_GLOBALS_STR is None:
        _register_safe_globals()
    assert _SAFE_GLOBALS_STR is not None
    return _SAFE_GLOBALS_STR


def _register_safe_globals() -> None:
    """Register safe types with both torch.serialization and build string allowlist.

    This function is called lazily (first time the allowlist is needed) to avoid
    import-time side effects that could trigger circular imports.

    The base set of types (gymnasium, numpy, cloudpickle, etc.) is collected here.
    SB3-specific types (policies, buffers, extractors, VecNormalize) must be
    registered separately via ``register_sb3_safe_globals()`` because those modules
    import from this file and we cannot import them at module load time.
    """
    global _SAFE_GLOBALS_STR, _TORCH_REGISTRATION_DONE

    if _TORCH_REGISTRATION_DONE:
        return

    base_types = _collect_base_types()

    # Build string allowlist for _RestrictedUnpickler
    _SAFE_GLOBALS_STR = set()
    for t in base_types:
        if hasattr(t, "__module__") and hasattr(t, "__qualname__"):
            _SAFE_GLOBALS_STR.add(f"{t.__module__}.{t.__qualname__}")
        else:
            # Some objects (like numpy random functions) may not have these
            try:
                _SAFE_GLOBALS_STR.add(f"{type(t).__module__}.{t.__name__}")
            except AttributeError:
                pass

    # Numpy-internal pickle helpers: both numpy 1.x (numpy.core.*) and
    # numpy 2.x (numpy._core.*) paths, plus random pickle helpers.
    for _name in _NUMPY_PICKLE_INTERNALS_STR:
        _SAFE_GLOBALS_STR.add(_name)

    # Explicit string entries for builtins and cloudpickle internals (these are
    # functions/objects whose module/qualname doesn't match what pickle embeds)
    for _name in (
        "builtins.tuple",
        "builtins.dict",
        "builtins.list",
        "builtins.set",
        "builtins.frozenset",
        "builtins.str",
        "builtins.int",
        "builtins.float",
        "builtins.bool",
        "builtins.bytes",
        "builtins.bytearray",
        "builtins.object",
        "builtins.type",
        "cloudpickle.cloudpickle",
        "cloudpickle.cloudpickle.__newobj__",
        "cloudpickle.cloudpickle._make_skeleton_function",
        "cloudpickle.cloudpickle._make_stepfunc",
        "cloudpickle.cloudpickle._make_fileless_lambda",
        "cloudpickle.cloudpickle.make_dict_fromnamedtuple",
        "cloudpickle.cloudpickle.make_dict_fromnamedtuple_with_defaults",
        "cloudpickle.cloudpickle.make_dynamic_classlookup",
        "cloudpickle.cloudpickle.make_function_from_globals",
        "cloudpickle.cloudpickle.make_instance_from_reduce",
        "cloudpickle.cloudpickle.make_local_from_global",
        "cloudpickle.cloudpickle.make_numpy_array",
        "cloudpickle.cloudpickle.make_numpy_scalar",
        "cloudpickle.cloudpickle.make_opaque_object",
        "cloudpickle.cloudpickle.make_object_from_newargs",
        "cloudpickle.cloudpickle.make_object_from_newargsreduce",
        "cloudpickle.cloudpickle.make_repr_from_name",
        "cloudpickle.cloudpickle.make_seq",
        "cloudpickle.cloudpickle.make_set",
        "cloudpickle.cloudpickle.make_skeleton_class",
        "cloudpickle.cloudpickle.make_skeleton_enum",
        "cloudpickle.cloudpickle.make_super",
        "cloudpickle.cloudpickle.make_type_var",
        "cloudpickle.cloudpickle.make_type_var_tuple",
        "cloudpickle.cloudpickle.make_typed_dict",
        "cloudpickle.cloudpickle.make_unordered_set",
        "cloudpickle.cloudpickle.restore_class",
        "cloudpickle.cloudpickle.restore_class_attr_descriptors",
        "cloudpickle.cloudpickle.restore_function",
        "cloudpickle.cloudpickle._class_setstate",
        "cloudpickle.cloudpickle._fillvar",
        "cloudpickle.cloudpickle.subimport",
        "cloudpickle.cloudpickle._lookup_module_and_obj_in_qualname",
        "cloudpickle.cloudpickle.whichmodule",
        "types.FunctionType",
        "types.ModuleType",
        "types.CellType",
        "copyreg.__newobj_ex__",
        "copyreg.__newobj__",
        "copyreg._reconstruct",
        "copyreg.reconstructor",
        "copyreg._reduce_ex",
    ):
        _SAFE_GLOBALS_STR.add(_name)

    # Register with torch.serialization
    th.serialization.add_safe_globals(base_types)  # type: ignore[arg-type]
    _TORCH_REGISTRATION_DONE = True


def register_sb3_safe_globals() -> None:
    """Register SB3-specific types (policies, buffers, extractors, VecNormalize).

    This must be called after SB3 modules are fully loaded (no circular import).
    It is called lazily from ``policies.py`` when policy loading actually happens.
    """
    global _SAFE_GLOBALS_STR

    # Ensure base registration is done first
    if not _TORCH_REGISTRATION_DONE:
        _register_safe_globals()

    # We can skip if already registered (check for a marker)
    if _SAFE_GLOBALS_STR is not None and "stable_baselines3.common.buffers.ReplayBuffer" in _SAFE_GLOBALS_STR:
        return

    from stable_baselines3.a2c.policies import (
        ActorCriticPolicy as A2CActorCriticPolicy,
    )
    from stable_baselines3.a2c.policies import (
        CnnPolicy as A2CCnnPolicy,
    )
    from stable_baselines3.a2c.policies import (
        MlpPolicy as A2CMlpPolicy,
    )
    from stable_baselines3.a2c.policies import (
        MultiInputPolicy as A2CMultiInputPolicy,
    )
    from stable_baselines3.common.buffers import (
        DictReplayBuffer,
        ReplayBuffer,
        RolloutBuffer,
    )
    from stable_baselines3.common.distributions import (
        BernoulliDistribution,
        CategoricalDistribution,
        DiagGaussianDistribution,
        MultiCategoricalDistribution,
        StateDependentNoiseDistribution,
    )
    from stable_baselines3.common.noise import (
        ActionNoise,
        NormalActionNoise,
        OrnsteinUhlenbeckActionNoise,
        VectorizedActionNoise,
    )
    from stable_baselines3.common.policies import (
        ActorCriticCnnPolicy,
        ActorCriticPolicy,
        BasePolicy,
        ContinuousCritic,
        MultiInputActorCriticPolicy,
    )
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    from stable_baselines3.common.torch_layers import (
        BaseFeaturesExtractor,
        CombinedExtractor,
        FlattenExtractor,
        MlpExtractor,
        NatureCNN,
    )
    from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
    from stable_baselines3.common.utils import (
        ConstantSchedule,
        FloatSchedule,
        LinearSchedule,
    )
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
    from stable_baselines3.ddpg.policies import (
        CnnPolicy as DdpgCnnPolicy,
    )
    from stable_baselines3.ddpg.policies import (
        MlpPolicy as DdpgMlpPolicy,
    )
    from stable_baselines3.ddpg.policies import (
        MultiInputPolicy as DdpgMultiInputPolicy,
    )
    from stable_baselines3.dqn.policies import (
        CnnPolicy as DqnCnnPolicy,
    )
    from stable_baselines3.dqn.policies import (
        MlpPolicy as DqnMlpPolicy,
    )
    from stable_baselines3.dqn.policies import (
        MultiInputPolicy as DqnMultiInputPolicy,
    )
    from stable_baselines3.dqn.policies import (
        QNetwork,
    )
    from stable_baselines3.ppo.policies import (
        ActorCriticPolicy as PpoActorCriticPolicy,
    )
    from stable_baselines3.ppo.policies import (
        CnnPolicy as PpoCnnPolicy,
    )
    from stable_baselines3.ppo.policies import (
        MlpPolicy as PpoMlpPolicy,
    )
    from stable_baselines3.ppo.policies import (
        MultiInputPolicy as PpoMultiInputPolicy,
    )
    from stable_baselines3.sac.policies import (
        CnnPolicy as SacCnnPolicy,
    )
    from stable_baselines3.sac.policies import (
        MlpPolicy as SacMlpPolicy,
    )
    from stable_baselines3.sac.policies import (
        MultiInputPolicy as SacMultiInputPolicy,
    )
    from stable_baselines3.td3.policies import (
        CnnPolicy as Td3CnnPolicy,
    )
    from stable_baselines3.td3.policies import (
        MlpPolicy as Td3MlpPolicy,
    )
    from stable_baselines3.td3.policies import (
        MultiInputPolicy as Td3MultiInputPolicy,
    )

    sb3_types = [
        # Buffers
        ReplayBuffer,
        DictReplayBuffer,
        RolloutBuffer,
        # Schedules
        FloatSchedule,
        ConstantSchedule,
        LinearSchedule,
        TrainFreq,
        TrainFrequencyUnit,
        # Action noise
        ActionNoise,
        NormalActionNoise,
        OrnsteinUhlenbeckActionNoise,
        VectorizedActionNoise,
        # Extractors
        FlattenExtractor,
        NatureCNN,
        CombinedExtractor,
        MlpExtractor,
        BaseFeaturesExtractor,
        # Distributions
        BernoulliDistribution,
        CategoricalDistribution,
        DiagGaussianDistribution,
        MultiCategoricalDistribution,
        StateDependentNoiseDistribution,
        # Base policy classes
        BasePolicy,
        ActorCriticPolicy,
        ActorCriticCnnPolicy,
        MultiInputActorCriticPolicy,
        ContinuousCritic,
        # Algo-specific policies
        A2CActorCriticPolicy,
        A2CCnnPolicy,
        A2CMlpPolicy,
        A2CMultiInputPolicy,
        PpoActorCriticPolicy,
        PpoCnnPolicy,
        PpoMlpPolicy,
        PpoMultiInputPolicy,
        DdpgCnnPolicy,
        DdpgMlpPolicy,
        DdpgMultiInputPolicy,
        DqnCnnPolicy,
        DqnMlpPolicy,
        DqnMultiInputPolicy,
        QNetwork,
        SacCnnPolicy,
        SacMlpPolicy,
        SacMultiInputPolicy,
        Td3CnnPolicy,
        Td3MlpPolicy,
        Td3MultiInputPolicy,
        # VecNormalize and running stats
        RunningMeanStd,
        VecNormalize,
    ]

    # Add to string allowlist
    for t in sb3_types:
        if _SAFE_GLOBALS_STR is not None and hasattr(t, "__module__") and hasattr(t, "__qualname__"):
            _SAFE_GLOBALS_STR.add(f"{t.__module__}.{t.__qualname__}")

    # Register with torch.serialization
    th.serialization.add_safe_globals(sb3_types)  # type: ignore[arg-type]


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows globals from a predefined allowlist.

    Blocks cloudpickle payloads that contain arbitrary code execution
    while still permitting the types that SB3 legitimately serialises.
    """

    def find_class(self, module: str, name: str) -> Any:
        global_full = f"{module}.{name}"
        if global_full in get_safe_globals():
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Global {global_full!r} is not in the safe deserialization allowlist. "
            "This is likely an attempt to execute arbitrary code via a crafted "
            "checkpoint. Use deserialization_mode='legacy' if you trust this file, "
            "or report the type to the SB3 maintainers."
        )


# ---------------------------------------------------------------------------
# Public API: extend the safe-globals allowlist (a la torch.serialization)
# ---------------------------------------------------------------------------

_USER_SAFE_GLOBALS: set[str] = set()


def add_safe_globals(
    safe_globals: list[type | tuple[type, str]] | type | tuple[type, str],
) -> None:
    """Register one or more classes/functions as safe for ``deserialization_mode="safe"``.

    This is the SB3 equivalent of :func:`torch.serialization.add_safe_globals`.
    Types registered here will be permitted by the restricted unpickler when
    ``deserialization_mode="safe"`` is used, for *all* subsequent loads in the
    current process.

    Each item can be:

    * A class or function object.  Its fully-qualified name
      ``module.qualname`` will be used automatically.
    * A ``(class_or_function, "explicit.module.Path")`` tuple when the
      pickle payload uses a different module path than the live object
      (e.g., the checkpoint was saved on a machine with a different
      package name).

    **Only register types you trust.**  The security guarantee of
    ``deserialization_mode="safe"`` is that *only* allowlisted globals can
    be invoked during unpickling.

    **Example**

    .. code-block:: python

       from stable_baselines3.common.safe_globals import add_safe_globals

       class MyCustomSpace(gymnasium.spaces.Space):
           ...

       add_safe_globals([MyCustomSpace])
       model = PPO.load("checkpoint.zip", deserialization_mode="safe")
    """
    if not isinstance(safe_globals, list):
        safe_globals = [safe_globals]

    for item in safe_globals:
        if isinstance(item, tuple):
            _, explicit_path = item
            _USER_SAFE_GLOBALS.add(explicit_path)
        else:
            _USER_SAFE_GLOBALS.add(f"{item.__module__}.{item.__qualname__}")


class safe_globals:
    """Context-manager that temporarily adds globals to the safe allowlist.

    The added types are automatically removed when the block exits.

    **Example**

    .. code-block:: python

       from stable_baselines3.common.safe_globals import safe_globals

       with safe_globals([MyCustomSpace]):
           model = PPO.load("checkpoint.zip", deserialization_mode="safe")
    """

    def __init__(
        self,
        safe_globals: list[type | tuple[type, str]] | type | tuple[type, str],
    ) -> None:
        self._items = safe_globals if isinstance(safe_globals, list) else [safe_globals]

    def __enter__(self) -> safe_globals:
        self._backup = _USER_SAFE_GLOBALS.copy()
        add_safe_globals(self._items)
        return self

    def __exit__(self, *args) -> None:
        _USER_SAFE_GLOBALS.clear()
        _USER_SAFE_GLOBALS.update(self._backup)


def get_safe_globals() -> set[str]:
    """Return the complete allowlist (built-in + user-registered) for safe deserialization."""
    base = _get_safe_globals_str()
    return base | _USER_SAFE_GLOBALS
