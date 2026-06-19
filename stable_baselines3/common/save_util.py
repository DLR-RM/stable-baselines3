"""
Save util taken from stable_baselines
used to serialize data (class parameters) of model classes
"""

import base64
import functools
import io
import json
import os
import pathlib
import pickle
import warnings
import zipfile
from typing import Any

import cloudpickle
import torch as th

import stable_baselines3 as sb3
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device, get_system_info

# ---------------------------------------------------------------------------
# Restricted unpickler for safe deserialization
# ---------------------------------------------------------------------------
#
# Cloudpickle serialises objects into a byte stream whose deserialization
# invokes ``find_class(module, name)`` for each global referenced.  By
# overriding ``find_class`` we can block arbitrary code execution while
# still allowing the small set of types that SB3 legitimately serialises.
#
# The allowlist is populated lazily to avoid circular imports (many SB3
# modules import save_util).
# ---------------------------------------------------------------------------

_SAFE_CLOUDPICKLE_GLOBALS: set[str] | None = None
_USER_SAFE_GLOBALS: set[str] = set()


def _get_safe_globals() -> set[str]:
    """Return the allowlist of safe globals, populating it lazily.

    Also includes any types registered by the user via ``add_safe_globals()``.
    """
    global _SAFE_CLOUDPICKLE_GLOBALS
    if _SAFE_CLOUDPICKLE_GLOBALS is None:
        _build_safe_globals()
    return _SAFE_CLOUDPICKLE_GLOBALS | _USER_SAFE_GLOBALS


def _build_safe_globals() -> None:
    """Populate _SAFE_CLOUDPICKLE_GLOBALS with built-in safe types."""
    global _SAFE_CLOUDPICKLE_GLOBALS
    _SAFE_CLOUDPICKLE_GLOBALS = set()

    # --- gymnasium spaces ---
    from gymnasium import spaces as gym_spaces

    for _cls in (
        gym_spaces.Box,
        gym_spaces.Discrete,
        gym_spaces.MultiBinary,
        gym_spaces.MultiDiscrete,
        gym_spaces.Dict,
        gym_spaces.Tuple,
        gym_spaces.Space,
    ):
        _SAFE_CLOUDPICKLE_GLOBALS.add(f"{_cls.__module__}.{_cls.__qualname__}")

    # --- old gym spaces (models saved with gym <= 0.26) ---
    try:
        import gym
        for _cls in (
            gym.spaces.Box,
            gym.spaces.Discrete,
            gym.spaces.MultiBinary,
            gym.spaces.MultiDiscrete,
            gym.spaces.Dict,
            gym.spaces.Tuple,
            gym.spaces.Space,
        ):
            _SAFE_CLOUDPICKLE_GLOBALS.add(f"{_cls.__module__}.{_cls.__qualname__}")
    except ImportError:
        pass

    # --- SB3 utility classes ---
    from stable_baselines3.common.buffers import (
        DictReplayBuffer,
        ReplayBuffer,
        RolloutBuffer,
    )
    from stable_baselines3.common.type_aliases import TrainFreq
    from stable_baselines3.common.utils import (
        ConstantSchedule,
        FloatSchedule,
        LinearSchedule,
    )

    from stable_baselines3.common.type_aliases import TrainFrequencyUnit

    for _cls in (
        FloatSchedule,
        ConstantSchedule,
        LinearSchedule,
        TrainFreq,
        TrainFrequencyUnit,
        RolloutBuffer,
        ReplayBuffer,
        DictReplayBuffer,
    ):
        _SAFE_CLOUDPICKLE_GLOBALS.add(f"{_cls.__module__}.{_cls.__qualname__}")

    # --- SB3 policy and extractor classes ---
    from stable_baselines3.a2c.policies import (
        ActorCriticPolicy as A2CActorCriticPolicy,
        CnnPolicy as A2CCnnPolicy,
        MlpPolicy as A2CMlpPolicy,
        MultiInputPolicy as A2CMultiInputPolicy,
    )
    from stable_baselines3.common.policies import (
        ActorCriticCnnPolicy,
        ActorCriticPolicy,
        ContinuousCritic,
        MultiInputActorCriticPolicy,
    )
    from stable_baselines3.common.torch_layers import (
        BaseFeaturesExtractor,
        CombinedExtractor,
        FlattenExtractor,
        MlpExtractor,
        NatureCNN,
    )
    from stable_baselines3.ddpg.policies import (
        CnnPolicy as DdpgCnnPolicy,
        MlpPolicy as DdpgMlpPolicy,
        MultiInputPolicy as DdpgMultiInputPolicy,
    )
    from stable_baselines3.dqn.policies import (
        CnnPolicy as DqnCnnPolicy,
        MlpPolicy as DqnMlpPolicy,
        MultiInputPolicy as DqnMultiInputPolicy,
        QNetwork,
    )
    from stable_baselines3.ppo.policies import (
        ActorCriticPolicy as PpoActorCriticPolicy,
        CnnPolicy as PpoCnnPolicy,
        MlpPolicy as PpoMlpPolicy,
        MultiInputPolicy as PpoMultiInputPolicy,
    )
    from stable_baselines3.sac.policies import (
        CnnPolicy as SacCnnPolicy,
        MlpPolicy as SacMlpPolicy,
        MultiInputPolicy as SacMultiInputPolicy,
    )
    from stable_baselines3.td3.policies import (
        CnnPolicy as Td3CnnPolicy,
        MlpPolicy as Td3MlpPolicy,
        MultiInputPolicy as Td3MultiInputPolicy,
    )

    for _cls in (
        A2CActorCriticPolicy, A2CCnnPolicy, A2CMlpPolicy, A2CMultiInputPolicy,
        PpoActorCriticPolicy, PpoCnnPolicy, PpoMlpPolicy, PpoMultiInputPolicy,
        DdpgCnnPolicy, DdpgMlpPolicy, DdpgMultiInputPolicy,
        DqnCnnPolicy, DqnMlpPolicy, DqnMultiInputPolicy, QNetwork,
        SacCnnPolicy, SacMlpPolicy, SacMultiInputPolicy,
        Td3CnnPolicy, Td3MlpPolicy, Td3MultiInputPolicy,
        ActorCriticPolicy, ActorCriticCnnPolicy, MultiInputActorCriticPolicy,
        ContinuousCritic,
        BaseFeaturesExtractor, CombinedExtractor, FlattenExtractor,
        MlpExtractor, NatureCNN,
    ):
        _SAFE_CLOUDPICKLE_GLOBALS.add(f"{_cls.__module__}.{_cls.__qualname__}")

    # --- numpy internals ---
    for _name in (
        "numpy.core.multiarray._reconstruct",
        "numpy.core.multiarray.scalar",
        "numpy._core.multiarray._reconstruct",
        "numpy._core.multiarray.scalar",
        "numpy._core.numeric._frombuffer",
        "numpy.dtype",
        "numpy.ndarray",
        "numpy.float32", "numpy.float64",
        "numpy.int32", "numpy.int64",
        "numpy.bool_",
        "numpy.dtypes.Float32DType", "numpy.dtypes.Float64DType",
        "numpy.dtypes.Int32DType", "numpy.dtypes.Int64DType",
        "numpy.dtypes.BoolDType",
        "numpy.random._pickle.__generator_ctor",
        "numpy.random._pickle.__bit_generator_ctor",
        "numpy.random._pickle.__randomstate_ctor",
        "numpy.random._pickle.GeneratorState",
        "numpy.random.bit_generator.BitGenerator",
        "numpy.random.bit_generator.SeedSequence",
        "numpy.random.bit_generator.__pyx_unpickle_SeedSequence",
        "numpy.random.bit_generator.__pyx_unpickle_SeedlessSeedSequence",
        "numpy.random._pcg64.PCG64",
        "numpy.random._mt19937.MT19937",
        "numpy.random._philox.Philox",
        "numpy.random._sfc64.SFC64",
        "numpy.random._generator.Generator",
        "numpy.random.mtrand.RandomState",
    ):
        _SAFE_CLOUDPICKLE_GLOBALS.add(_name)

    # --- PyTorch types used by policies and buffers ---
    _SAFE_CLOUDPICKLE_GLOBALS.add("torch.optim.rmsprop.RMSprop")
    _SAFE_CLOUDPICKLE_GLOBALS.add("torch.device")

    # --- SB3 VecNormalize and running stats ---
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
    for _cls in (RunningMeanStd, VecNormalize):
        _SAFE_CLOUDPICKLE_GLOBALS.add(f"{_cls.__module__}.{_cls.__qualname__}")

    # --- Python builtins ---
    for _name in (
        "builtins.tuple", "builtins.dict", "builtins.list", "builtins.set",
        "builtins.frozenset", "builtins.str", "builtins.int", "builtins.float",
        "builtins.bool", "builtins.bytes", "builtins.bytearray",
        "builtins.object", "builtins.type",
    ):
        _SAFE_CLOUDPICKLE_GLOBALS.add(_name)

    # --- cloudpickle internals ---
    # cloudpickle 3.x uses different internal names than 2.x.
    # We allow the full set of known cloudpickle helpers so that
    # checkpoints saved with either major version can be loaded.
    for _name in (
        "cloudpickle.cloudpickle",
        "cloudpickle.cloudpickle.__newobj__",
        "cloudpickle.cloudpickle._make_skeleton_class",
        "cloudpickle.cloudpickle._make_skeleton_enum",
        "cloudpickle.cloudpickle._make_skeleton_function",
        "cloudpickle.cloudpickle._make_cell",
        "cloudpickle.cloudpickle._make_empty_cell",
        "cloudpickle.cloudpickle._make_stepfunc",
        "cloudpickle.cloudpickle._make_fileless_lambda",
        "cloudpickle.cloudpickle._make_function",
        "cloudpickle.cloudpickle._make_dict_items",
        "cloudpickle.cloudpickle._make_dict_keys",
        "cloudpickle.cloudpickle._make_dict_values",
        "cloudpickle.cloudpickle._make_typevar",
        "cloudpickle.cloudpickle._builtin_type",
        "cloudpickle.cloudpickle._function_setstate",
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
        _SAFE_CLOUDPICKLE_GLOBALS.add(_name)

    # --- Python standard-library containers ---
    for _name in (
        "collections.deque",
        "operator.getstate",
    ):
        _SAFE_CLOUDPICKLE_GLOBALS.add(_name)


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows globals from a predefined allowlist.

    Blocks cloudpickle payloads that contain arbitrary code execution
    while still permitting the types that SB3 legitimately serialises.
    """

    def find_class(self, module: str, name: str):
        global_full = f"{module}.{name}"
        if global_full in _get_safe_globals():
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Global {global_full!r} is not in the safe deserialization allowlist. "
            "This is likely an attempt to execute arbitrary code via a crafted "
            "checkpoint. Use deserialization_mode='legacy' if you trust this file, "
            "or report the type to the SB3 maintainers."
        )


def _cloudpickle_loads_safe(data: bytes) -> Any:
    """Deserialize cloudpickle data using the restricted allowlist unpickler."""
    return _RestrictedUnpickler(io.BytesIO(data)).load()


# ---------------------------------------------------------------------------
# Public API: extend the safe-globals allowlist (a la torch.serialization)
# ---------------------------------------------------------------------------


def add_safe_globals(
    safe_globals: list[type | tuple[type, str]] | type | tuple[type, str],
) -> None:
    """
    Register one or more classes/functions as safe for ``deserialization_mode='safe'``.

    This is the SB3 equivalent of :func:`torch.serialization.add_safe_globals`.
    Types registered here will be permitted by the restricted unpickler when
    ``deserialization_mode='safe'`` is used, for *all* subsequent loads in the
    current process.

    Each item can be:

    * A class or function object.  Its fully-qualified name
      ``module.qualname`` will be used automatically.
    * A ``(class_or_function, "explicit.module.Path")`` tuple when the
      pickle payload uses a different module path than the live object
      (e.g., the checkpoint was saved on a machine with a different
      package name).

    **Only register types you trust.**  The security guarantee of
    ``deserialization_mode='safe'`` is that *only* allowlisted globals can
    be invoked during unpickling.

    **Example**

    .. code-block:: python

       from stable_baselines3.common.save_util import add_safe_globals

       class MyCustomSpace(gymnasium.spaces.Space):
           ...

       add_safe_globals([MyCustomSpace])
       model = PPO.load("checkpoint.zip", deserialization_mode="safe")
    """
    if not isinstance(safe_globals, list):
        safe_globals = [safe_globals]

    for item in safe_globals:
        if isinstance(item, tuple):
            obj, explicit_path = item
            _USER_SAFE_GLOBALS.add(explicit_path)
        else:
            _USER_SAFE_GLOBALS.add(f"{item.__module__}.{item.__qualname__}")


class safe_globals:
    """Context-manager that temporarily adds globals to the safe allowlist.

    The added types are automatically removed when the block exits.

    **Example**

    .. code-block:: python

       from stable_baselines3.common.save_util import safe_globals

       with safe_globals([MyCustomSpace]):
           model = PPO.load("checkpoint.zip", deserialization_mode="safe")
    """

    def __init__(
        self,
        safe_globals: list[type | tuple[type, str]] | type | tuple[type, str],
    ) -> None:
        self._items = safe_globals if isinstance(safe_globals, list) else [safe_globals]

    def __enter__(self) -> "safe_globals":
        self._backup = _USER_SAFE_GLOBALS.copy()
        add_safe_globals(self._items)
        return self

    def __exit__(self, *args: Any) -> None:
        _USER_SAFE_GLOBALS.clear()
        _USER_SAFE_GLOBALS.update(self._backup)


def recursive_getattr(obj: Any, attr: str, *args) -> Any:
    """
    Recursive version of getattr
    taken from https://stackoverflow.com/questions/31174295

    Ex:
    > MyObject.sub_object = SubObject(name='test')
    > recursive_getattr(MyObject, 'sub_object.name')  # return test
    :param obj:
    :param attr: Attribute to retrieve
    :return: The attribute
    """

    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj, *attr.split(".")])


def recursive_setattr(obj: Any, attr: str, val: Any) -> None:
    """
    Recursive version of setattr
    taken from https://stackoverflow.com/questions/31174295

    Ex:
    > MyObject.sub_object = SubObject(name='test')
    > recursive_setattr(MyObject, 'sub_object.name', 'hello')
    :param obj:
    :param attr: Attribute to set
    :param val: New value of the attribute
    """
    pre, _, post = attr.rpartition(".")
    return setattr(recursive_getattr(obj, pre) if pre else obj, post, val)


def is_json_serializable(item: Any) -> bool:
    """
    Test if an object is serializable into JSON

    :param item: The object to be tested for JSON serialization.
    :return: True if object is JSON serializable, false otherwise.
    """
    # Try with try-except struct.
    json_serializable = True
    try:
        _ = json.dumps(item)
    except TypeError:
        json_serializable = False
    return json_serializable


def data_to_json(data: dict[str, Any]) -> str:
    """
    Turn data (class parameters) into a JSON string for storing

    :param data: Dictionary of class parameters to be
        stored. Items that are not JSON serializable will be
        pickled with Cloudpickle and stored as bytearray in
        the JSON file
    :return: JSON string of the data serialized.
    """
    # First, check what elements can not be JSONfied,
    # and turn them into byte-strings
    serializable_data = {}
    for data_key, data_item in data.items():
        # See if object is JSON serializable
        if is_json_serializable(data_item):
            # All good, store as it is
            serializable_data[data_key] = data_item
        else:
            # Not serializable, cloudpickle it into
            # bytes and convert to base64 string for storing.
            # Also store type of the class for consumption
            # from other languages/humans, so we have an
            # idea what was being stored.
            base64_encoded = base64.b64encode(cloudpickle.dumps(data_item)).decode()

            # Use ":" to make sure we do
            # not override these keys
            # when we include variables of the object later
            cloudpickle_serialization = {
                ":type:": str(type(data_item)),
                ":serialized:": base64_encoded,
            }

            # Add first-level JSON-serializable items of the
            # object for further details (but not deeper than this to
            # avoid deep nesting).
            # First we check that object has attributes (not all do,
            # e.g. numpy scalars)
            if hasattr(data_item, "__dict__") or isinstance(data_item, dict):
                # Take elements from __dict__ for custom classes
                item_generator = data_item.items if isinstance(data_item, dict) else data_item.__dict__.items
                for variable_name, variable_item in item_generator():
                    # Check if serializable. If not, just include the
                    # string-representation of the object.
                    if is_json_serializable(variable_item):
                        cloudpickle_serialization[variable_name] = variable_item
                    else:
                        cloudpickle_serialization[variable_name] = str(variable_item)

            serializable_data[data_key] = cloudpickle_serialization
    json_string = json.dumps(serializable_data, indent=4)
    return json_string


def json_to_data(
    json_string: str,
    custom_objects: dict[str, Any] | None = None,
    deserialization_mode: str = "safe",
) -> dict[str, Any]:
    """
    Turn JSON serialization of class-parameters back into dictionary.

    :param json_string: JSON serialization of the class-parameters
        that should be loaded.
    :param custom_objects: Dictionary of objects to replace
        upon loading. If a variable is present in this dictionary as a
        key, it will not be deserialized and the corresponding item
        will be used instead. Similar to custom_objects in
        ``keras.models.load_model``. Useful when you have an object in
        file that can not be deserialized.
    :param deserialization_mode: How to handle cloudpickle-serialized objects
        stored in the checkpoint's ``data`` JSON.

        - ``"safe"`` (default): Deserialize using a restricted unpickler that
          only allows a fixed allowlist of known-safe SB3/gymnasium/numpy types.
          Any cloudpickle payload referencing a type outside this allowlist is
          rejected with a clear error.  A single ``SecurityWarning`` is emitted.
        - ``"legacy"``: Deserialize all ``:serialized:`` entries with the
          unrestricted cloudpickle loader.  This preserves full backward
          compatibility but **executes arbitrary Python code** embedded in the
          checkpoint.
    :return: Loaded class parameters.
    """
    if custom_objects is not None and not isinstance(custom_objects, dict):
        raise ValueError("custom_objects argument must be a dict or None")

    if deserialization_mode not in ("legacy", "safe"):
        raise ValueError(
            f"deserialization_mode must be 'legacy' or 'safe', got {deserialization_mode!r}"
        )

    json_dict = json.loads(json_string)
    # This will be filled with deserialized data
    return_data = {}
    warned_once = False
    for data_key, data_item in json_dict.items():
        if custom_objects is not None and data_key in custom_objects.keys():
            # If item is provided in custom_objects, replace
            # the one from JSON with the one in custom_objects
            return_data[data_key] = custom_objects[data_key]
        elif isinstance(data_item, dict) and ":serialized:" in data_item.keys():
            # If item is dictionary with ":serialized:"
            # key, this means it is serialized with cloudpickle.
            if not warned_once:
                warned_once = True
                if deserialization_mode == "safe":
                    pass
                    # warnings.warn(
                    #     "Loading a model checkpoint that contains cloudpickle-serialized "
                    #     "objects with a restricted (safe) deserializer. Only known-safe "
                    #     "SB3/gymnasium/numpy types are allowed. "
                    #     UserWarning,
                    # )
                else:
                    warnings.warn(
                        "Loading a model checkpoint that contains cloudpickle-serialized "
                        "objects (deserialization_mode='legacy'). This allows arbitrary "
                        "Python code execution from the checkpoint file. Only load "
                        "checkpoints from trusted sources. To enable safe deserialization, "
                        "use deserialization_mode='safe'. ",
                        UserWarning,
                    )

            serialization = data_item[":serialized:"]
            try:
                base64_object = base64.b64decode(serialization.encode())
                if deserialization_mode == "safe":
                    deserialized_object = _cloudpickle_loads_safe(base64_object)
                else:
                    deserialized_object = cloudpickle.loads(base64_object)
            except (RuntimeError, TypeError, AttributeError, pickle.UnpicklingError) as e:
                warnings.warn(
                    f"Could not deserialize object {data_key}. "
                    "Consider using `custom_objects` argument to replace "
                    "this object.\n"
                    f"Exception: {e}"
                )
            else:
                return_data[data_key] = deserialized_object
        else:
            # Read as it is
            return_data[data_key] = data_item
    return return_data


@functools.singledispatch
def open_path(
    path: str | pathlib.Path | io.BufferedIOBase, mode: str, verbose: int = 0, suffix: str | None = None
) -> io.BufferedWriter | io.BufferedReader | io.BytesIO | io.BufferedRandom:
    """
    Opens a path for reading or writing with a preferred suffix and raises debug information.
    If the provided path is a derivative of io.BufferedIOBase it ensures that the file
    matches the provided mode, i.e. If the mode is read ("r", "read") it checks that the path is readable.
    If the mode is write ("w", "write") it checks that the file is writable.

    If the provided path is a string or a pathlib.Path, it ensures that it exists. If the mode is "read"
    it checks that it exists, if it doesn't exist it attempts to read path.suffix if a suffix is provided.
    If the mode is "write" and the path does not exist, it creates all the parent folders. If the path
    points to a folder, it changes the path to path_2. If the path already exists and verbose >= 2,
    it raises a warning.

    :param path: the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param mode: how to open the file. "w"|"write" for writing, "r"|"read" for reading.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param suffix: The preferred suffix. If mode is "w" then the opened file has the suffix.
        If mode is "r" then we attempt to open the path. If an error is raised and the suffix
        is not None, we attempt to open the path with the suffix.
    :return:
    """
    # Note(antonin): the true annotation should be IO[bytes]
    # but there is not easy way to check that
    allowed_types = (io.BufferedWriter, io.BufferedReader, io.BytesIO, io.BufferedRandom)
    if not isinstance(path, allowed_types):
        raise TypeError(f"Path {path} parameter has invalid type: expected one of {allowed_types}.")
    if path.closed:
        raise ValueError(f"File stream {path} is closed.")
    mode = mode.lower()
    try:
        mode = {"write": "w", "read": "r", "w": "w", "r": "r"}[mode]
    except KeyError as e:
        raise ValueError("Expected mode to be either 'w' or 'r'.") from e
    if (("w" == mode) and not path.writable()) or (("r" == mode) and not path.readable()):
        error_msg = "writable" if "w" == mode else "readable"
        raise ValueError(f"Expected a {error_msg} file.")
    return path


@open_path.register(str)
def open_path_str(path: str, mode: str, verbose: int = 0, suffix: str | None = None) -> io.BufferedIOBase:
    """
    Open a path given by a string. If writing to the path, the function ensures
    that the path exists.

    :param path: the path to open. If mode is "w" then it ensures that the path exists
        by creating the necessary folders and renaming path if it points to a folder.
    :param mode: how to open the file. "w" for writing, "r" for reading.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param suffix: The preferred suffix. If mode is "w" then the opened file has the suffix.
        If mode is "r" then we attempt to open the path. If an error is raised and the suffix
        is not None, we attempt to open the path with the suffix.
    :return:
    """
    return open_path_pathlib(pathlib.Path(path), mode, verbose, suffix)


@open_path.register(pathlib.Path)
def open_path_pathlib(path: pathlib.Path, mode: str, verbose: int = 0, suffix: str | None = None) -> io.BufferedIOBase:
    """
    Open a path given by a string. If writing to the path, the function ensures
    that the path exists.

    :param path: the path to check. If mode is "w" then it
        ensures that the path exists by creating the necessary folders and
        renaming path if it points to a folder.
    :param mode: how to open the file. "w" for writing, "r" for reading.
    :param verbose: Verbosity level: 0 for no output, 2 for indicating if path without suffix is not found when mode is "r"
    :param suffix: The preferred suffix. If mode is "w" then the opened file has the suffix.
        If mode is "r" then we attempt to open the path. If an error is raised and the suffix
        is not None, we attempt to open the path with the suffix.
    :return:
    """
    if mode not in ("w", "r"):
        raise ValueError("Expected mode to be either 'w' or 'r'.")

    if mode == "r":
        try:
            return open_path(path.open("rb"), mode, verbose, suffix)
        except FileNotFoundError as error:
            if suffix is not None and suffix != "":
                newpath = pathlib.Path(f"{path}.{suffix}")
                if verbose >= 2:
                    warnings.warn(f"Path '{path}' not found. Attempting {newpath}.")
                path, suffix = newpath, None
            else:
                raise error
    else:
        try:
            if path.suffix == "" and suffix is not None and suffix != "":
                path = pathlib.Path(f"{path}.{suffix}")
            if path.exists() and path.is_file() and verbose >= 2:
                warnings.warn(f"Path '{path}' exists, will overwrite it.")
            return open_path(path.open("wb"), mode, verbose, suffix)
        except IsADirectoryError:
            warnings.warn(f"Path '{path}' is a folder. Will save instead to {path}_2")
            path = pathlib.Path(f"{path}_2")
        except FileNotFoundError:  # Occurs when the parent folder doesn't exist
            warnings.warn(f"Path '{path.parent}' does not exist. Will create it.")
            path.parent.mkdir(exist_ok=True, parents=True)

    # if opening was successful uses the open_path() function
    # if opening failed with IsADirectory|FileNotFound, calls open_path_pathlib
    #   with corrections
    # if reading failed with FileNotFoundError, calls open_path_pathlib with suffix
    return open_path_pathlib(path, mode, verbose, suffix)


def save_to_zip_file(
    save_path: str | pathlib.Path | io.BufferedIOBase,
    data: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    pytorch_variables: dict[str, Any] | None = None,
    verbose: int = 0,
) -> None:
    """
    Save model data to a zip archive.

    :param save_path: Where to store the model.
        if save_path is a str or pathlib.Path ensures that the path actually exists.
    :param data: Class parameters being stored (non-PyTorch variables)
    :param params: Model parameters being stored expected to contain an entry for every
                   state_dict with its name and the state_dict.
    :param pytorch_variables: Other PyTorch variables expected to contain name and value of the variable.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    file = open_path(save_path, "w", verbose=0, suffix="zip")
    # data/params can be None, so do not
    # try to serialize them blindly
    if data is not None:
        serialized_data = data_to_json(data)

    # Create a zip-archive and write our objects there.
    with zipfile.ZipFile(file, mode="w") as archive:
        # Do not try to save "None" elements
        if data is not None:
            archive.writestr("data", serialized_data)
        if pytorch_variables is not None:
            with archive.open("pytorch_variables.pth", mode="w", force_zip64=True) as pytorch_variables_file:
                th.save(pytorch_variables, pytorch_variables_file)
        if params is not None:
            for file_name, dict_ in params.items():
                with archive.open(file_name + ".pth", mode="w", force_zip64=True) as param_file:
                    th.save(dict_, param_file)
        # Save metadata: library version when file was saved
        archive.writestr("_stable_baselines3_version", sb3.__version__)
        # Save system info about the current python env
        archive.writestr("system_info.txt", get_system_info(print_info=False)[1])

    if isinstance(save_path, (str, pathlib.Path)):
        file.close()


def save_to_pkl(path: str | pathlib.Path | io.BufferedIOBase, obj: Any, verbose: int = 0) -> None:
    """
    Save an object to path creating the necessary folders along the way.
    If the path exists and is a directory, it will raise a warning and rename the path.
    If a suffix is provided in the path, it will use that suffix, otherwise, it will use '.pkl'.

    :param path: the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param obj: The object to save.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    file = open_path(path, "w", verbose=verbose, suffix="pkl")
    # Use protocol>=4 to support saving replay buffers >= 4Gb
    # See https://docs.python.org/3/library/pickle.html
    pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    if isinstance(path, (str, pathlib.Path)):
        file.close()


def load_from_pkl(
    path: str | pathlib.Path | io.BufferedIOBase,
    verbose: int = 0,
    deserialization_mode: str = "legacy",
) -> Any:
    """
    Load an object from the path. If a suffix is provided in the path, it will use that suffix.
    If the path does not exist, it will attempt to load using the .pkl suffix.

    :param path: the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param deserialization_mode: How to handle pickle deserialization.

        - ``"legacy"`` (default): Deserialize with ``pickle.load()``.  This preserves
          backward compatibility but **executes arbitrary Python code** embedded in
          the pickle file.  A ``SecurityWarning`` is emitted.
        - ``"safe"``: Deserialize using a restricted unpickler that only allows a
          fixed allowlist of known-safe SB3/gymnasium/numpy types.  Any pickle
          payload referencing a type outside this allowlist is rejected with a
          clear error.
    """
    if deserialization_mode not in ("legacy", "safe"):
        raise ValueError(
            f"deserialization_mode must be 'legacy' or 'safe', got {deserialization_mode!r}"
        )

    file = open_path(path, "r", verbose=verbose, suffix="pkl")

    if deserialization_mode == "safe":
        warnings.warn(
            "Loading a .pkl file with a restricted (safe) deserializer. Only known-safe "
            "SB3/gymnasium/numpy types are allowed. ",
            UserWarning,
        )
        obj = _RestrictedUnpickler(file).load()
    else:
        warnings.warn(
            "Loading a .pkl file with pickle deserialization (deserialization_mode='legacy'). "
            "This can execute arbitrary Python code from the file. Only load pickle files "
            "from trusted sources. ",
            UserWarning,
        )
        obj = pickle.load(file)
    if isinstance(path, (str, pathlib.Path)):
        file.close()
    return obj


def load_from_zip_file(
    load_path: str | pathlib.Path | io.BufferedIOBase,
    load_data: bool = True,
    custom_objects: dict[str, Any] | None = None,
    device: th.device | str = "auto",
    verbose: int = 0,
    print_system_info: bool = False,
    deserialization_mode: str = "legacy",
) -> tuple[dict[str, Any] | None, TensorDict, TensorDict | None]:
    """
    Load model data from a .zip archive

    :param load_path: Where to load the model from
    :param load_data: Whether we should load and return data
        (class parameters). Mainly used by 'load_parameters' to only load model parameters (weights)
    :param custom_objects: Dictionary of objects to replace
        upon loading. If a variable is present in this dictionary as a
        key, it will not be deserialized and the corresponding item
        will be used instead. Similar to custom_objects in
        ``keras.models.load_model``. Useful when you have an object in
        file that can not be deserialized.
    :param device: Device on which the code should run.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param print_system_info: Whether to print or not the system info
        about the saved model.
    :param deserialization_mode: How to handle cloudpickle-serialized objects
        in the checkpoint's ``data`` JSON.  See :func:`json_to_data` for details.
        Default is ``"legacy"`` for backward compatibility.
    :return: Class parameters, model state_dicts (aka "params", dict of state_dict)
        and dict of pytorch variables
    """
    file = open_path(load_path, "r", verbose=verbose, suffix="zip")

    # set device to cpu if cuda is not available
    device = get_device(device=device)

    # Open the zip archive and load data
    try:
        with zipfile.ZipFile(file) as archive:
            namelist = archive.namelist()
            # If data or parameters is not in the
            # zip archive, assume they were stored
            # as None (_save_to_file_zip allows this).
            data = None
            pytorch_variables = None
            params = {}

            # Debug system info first
            if print_system_info:
                if "system_info.txt" in namelist:
                    print("== SAVED MODEL SYSTEM INFO ==")
                    print(archive.read("system_info.txt").decode())
                else:
                    warnings.warn(
                        "The model was saved with SB3 <= 1.2.0 and thus cannot print system information.",
                        UserWarning,
                    )

            if "data" in namelist and load_data:
                # Load class parameters that are stored
                # with either JSON or pickle (not PyTorch variables).
                json_data = archive.read("data").decode()
                data = json_to_data(
                    json_data,
                    custom_objects=custom_objects,
                    deserialization_mode=deserialization_mode,
                )

            # Check for all .pth files and load them using th.load.
            # "pytorch_variables.pth" stores PyTorch variables, and any other .pth
            # files store state_dicts of variables with custom names (e.g. policy, policy.optimizer)
            pth_files = [file_name for file_name in namelist if os.path.splitext(file_name)[1] == ".pth"]
            for file_path in pth_files:
                with archive.open(file_path, mode="r") as param_file:
                    th_object = th.load(param_file, map_location=device, weights_only=True)
                    # "tensors.pth" was renamed "pytorch_variables.pth" in v0.9.0, see PR #138
                    if file_path == "pytorch_variables.pth" or file_path == "tensors.pth":
                        # PyTorch variables (not state_dicts)
                        pytorch_variables = th_object
                    else:
                        # State dicts. Store into params dictionary
                        # with same name as in .zip file (without .pth)
                        params[os.path.splitext(file_path)[0]] = th_object
    except zipfile.BadZipFile as e:
        # load_path wasn't a zip file
        raise ValueError(f"Error: the file {load_path} wasn't a zip-file") from e
    finally:
        if isinstance(load_path, (str, pathlib.Path)):
            file.close()
    return data, params, pytorch_variables
