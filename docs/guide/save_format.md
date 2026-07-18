(save-format)=

# On saving and loading

Stable Baselines3 (SB3) stores both neural network parameters and algorithm-related parameters such as
exploration schedule, number of environments and observation/action space. This allows continual learning and easy
use of trained agents without training, but it is not without its issues. Following describes the format
used to save agents in SB3 along with its pros and shortcomings.

Terminology used in this page:

- *parameters* refer to neural network parameters (also called "weights"). This is a dictionary
  mapping variable name to a PyTorch tensor.
- *data* refers to RL algorithm parameters, e.g. learning rate, exploration schedule, action/observation space.
  These depend on the algorithm used. This is a dictionary mapping classes variable names to their values.

## Zip-archive

A zip-archived JSON dump, PyTorch state dictionaries and PyTorch variables. The data dictionary (class parameters)
is stored as a JSON file, model parameters and optimizers are serialized with `torch.save()` function and these files
are stored under a single .zip archive.

Any objects that are not JSON serializable are serialized with cloudpickle and stored as base64-encoded
string in the JSON file, along with some information that was stored in the serialization. This allows
inspecting stored objects without deserializing the object itself.

This format allows skipping elements in the file, i.e. we can skip deserializing objects that are
broken/non-serializable.
This can be done via `custom_objects` argument to load functions.

:::{note}
If you encounter loading issue, for instance pickle issues or error after loading
(see [#171](https://github.com/DLR-RM/stable-baselines3/issues/171) or [#573](https://github.com/DLR-RM/stable-baselines3/issues/573)),
you can pass `print_system_info=True`
to compare the system on which the model was trained vs the current one
`model = PPO.load("ppo_saved", print_system_info=True)`
:::

File structure:

```
saved_model.zip/
├── data              JSON file of class-parameters (dictionary)
├── *.optimizer.pth   PyTorch optimizers serialized
├── policy.pth        PyTorch state dictionary of the policy saved
├── pytorch_variables.pth Additional PyTorch variables
├── _stable_baselines3_version contains the SB3 version with which the model was saved
├── system_info.txt contains system info (os, python version, ...) on which the model was saved
```

Pros:

- More robust to unserializable objects (one bad object does not break everything).
- Saved files can be inspected/extracted with zip-archive explorers and by other languages.

Cons:

- More complex implementation.
- Still relies partly on cloudpickle for complex objects (e.g. custom functions)
  with can lead to [incompatibilities](https://github.com/DLR-RM/stable-baselines3/issues/172) between Python versions.

## Secure Deserialization

:::{warning}
**Loading untrusted checkpoints can execute arbitrary Python code.**
Starting with SB3 2.10, all `load()` methods use `deserialization_mode="safe"` by default, which blocks
arbitrary code execution during deserialization at the cost of skipping non-whitelisted serialized entries.
:::

The `deserialization_mode` parameter is available on all load methods:

- `stable_baselines3.common.base_class.BaseAlgorithm.load`
- `stable_baselines3.common.save_util.json_to_data`
- `stable_baselines3.common.save_util.load_from_pkl`
- `stable_baselines3.common.save_util.load_from_zip_file`
- `stable_baselines3.common.off_policy_algorithm.OffPolicyAlgorithm.load_replay_buffer`
- `stable_baselines3.common.vec_env.vec_normalize.VecNormalize.load`

Each accepts one of two modes:

### Safe mode (default)

In `deserialization_mode="safe"`, SB3 uses a restricted unpickler that only allows a fixed allowlist of
known-safe types (SB3 classes, gymnasium spaces, numpy types, PyTorch types, cloudpickle internals).
If a serialized entry references a type outside the allowlist, it throws an error (that is caught), and the
user must supply a safe replacement via the `custom_objects` argument or extends the whitelist (see below).

```python
from stable_baselines3 import PPO

# If the checkpoint contains a custom learning-rate schedule that is not
# in the allowlist, you must provide it via custom_objects:
loaded = PPO.load(
    "model.zip",
    custom_objects={
        "learning_rate": 0.0003,
        "lr_schedule": lambda progress: progress * 0.0003,
    },
)
```

### Legacy mode

In `deserialization_mode="legacy"`, SB3 falls back to the standard `cloudpickle` / `pickle` loader.
This preserves full backward compatibility with models saved before SB3 2.10 but **may execute arbitrary
Python code** embedded in the checkpoint. A `UserWarning` is emitted.

```python
# Restores the pre-2.10 loading behavior for checkpoints that contain
# lambda functions, local classes, or custom gym environments:
loaded = PPO.load("model.zip", deserialization_mode="legacy")

# If you only load models from trusted sources, you can silence the warning with:
# import warnings
# warnings.filterwarnings("ignore", message="Loading a model checkpoint that contains cloudpickle-serialized objects", category=UserWarning)
```

### Extending the Safe Allowlist

If you have custom types (e.g. a custom environment or a custom space) that are safe to deserialize,
you can register them with the allowlist using `stable_baselines3.common.safe_globals.add_safe_globals`:

```python
from stable_baselines3.common.safe_globals import add_safe_globals
from my_module import MyCustomSpace

add_safe_globals(MyCustomSpace)
loaded = PPO.load("model.zip", deserialization_mode="safe")
```

For a temporary, scope-limited registration, use the `stable_baselines3.common.safe_globals.safe_globals`
context manager:

```python
from stable_baselines3.common.safe_globals import SafeGlobals
from my_module import MyCustomSpace

with SafeGlobals(MyCustomSpace):
    loaded = PPO.load("model.zip", deserialization_mode="safe")
# MyCustomSpace is automatically removed from the allowlist on exit
```

:::{note}
With the default `deserialization_mode="safe"`, you may encounter a `pickle.UnpicklingError`  or `Could not deserialize object` warning when loading checkpoints containing custom types that are not in the allowlist. The error message 
will indicate the missing type (e.g., `Global 'my_module.MyCustomType' is not in the safe deserialization allowlist`).
You can either use `add_safe_globals()` or `SafeGlobals` to register your custom types, pass `custom_objects=...` at load time, or switch to `deserialization_mode="legacy"` 
if you trust the checkpoint source.
:::
