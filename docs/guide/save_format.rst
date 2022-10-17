.. _save_format:


On saving and loading
=====================

Stable Baselines3 (SB3) stores both neural network parameters and algorithm-related parameters such as
exploration schedule, number of environments and observation/action space. This allows continual learning and easy
use of trained agents without training, but it is not without its issues. Following describes the format
used to save agents in SB3 along with its pros and shortcomings.

Terminology used in this page:

-  *parameters* refer to neural network parameters (also called "weights"). This is a dictionary
   mapping variable name to a PyTorch tensor.
-  *data* refers to RL algorithm parameters, e.g. learning rate, exploration schedule, action/observation space.
   These depend on the algorithm used. This is a dictionary mapping classes variable names to their values.


Zip-archive
-----------

A zip-archived JSON dump, PyTorch state dictionaries and PyTorch variables. The data dictionary (class parameters)
is stored as a JSON file, model parameters and optimizers are serialized with ``torch.save()`` function and these files
are stored under a single .zip archive.

Any objects that are not JSON serializable are serialized with cloudpickle and stored as base64-encoded
string in the JSON file, along with some information that was stored in the serialization. This allows
inspecting stored objects without deserializing the object itself.

This format allows skipping elements in the file, i.e. we can skip deserializing objects that are
broken/non-serializable.
This can be done via ``custom_objects`` argument to load functions.

.. note::

  If you encounter loading issue, for instance pickle issues or error after loading
  (see `#171 <https://github.com/DLR-RM/stable-baselines3/issues/171>`_ or `#573 <https://github.com/DLR-RM/stable-baselines3/issues/573>`_),
  you can pass ``print_system_info=True``
  to compare the system on which the model was trained vs the current one
  ``model = PPO.load("ppo_saved", print_system_info=True)``


File structure:

::

  saved_model.zip/
  ├── data              JSON file of class-parameters (dictionary)
  ├── *.optimizer.pth   PyTorch optimizers serialized
  ├── policy.pth        PyTorch state dictionary of the policy saved
  ├── pytorch_variables.pth Additional PyTorch variables
  ├── _stable_baselines3_version contains the SB3 version with which the model was saved
  ├── system_info.txt contains system info (os, python version, ...) on which the model was saved


Pros:

- More robust to unserializable objects (one bad object does not break everything).
- Saved files can be inspected/extracted with zip-archive explorers and by other languages.


Cons:

- More complex implementation.
- Still relies partly on cloudpickle for complex objects (e.g. custom functions)
  with can lead to `incompatibilities <https://github.com/DLR-RM/stable-baselines3/issues/172>`_ between Python versions.
