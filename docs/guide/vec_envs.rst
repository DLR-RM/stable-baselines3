.. _vec_env:

.. automodule:: stable_baselines3.common.vec_env

Vectorized Environments
=======================

Vectorized Environments are a method for stacking multiple independent environments into a single environment.
Instead of training an RL agent on 1 environment per step, it allows us to train it on ``n`` environments per step.
Because of this, ``actions`` passed to the environment are now a vector (of dimension ``n``).
It is the same for ``observations``, ``rewards`` and end of episode signals (``dones``).
In the case of non-array observation spaces such as ``Dict`` or ``Tuple``, where different sub-spaces
may have different shapes, the sub-observations are vectors (of dimension ``n``).

============= ======= ============ ======== ========= ================
Name          ``Box`` ``Discrete`` ``Dict`` ``Tuple`` Multi Processing
============= ======= ============ ======== ========= ================
DummyVecEnv   ✔️       ✔️           ✔️        ✔️         ❌️
SubprocVecEnv ✔️       ✔️           ✔️        ✔️         ✔️
============= ======= ============ ======== ========= ================

.. note::

	Vectorized environments are required when using wrappers for frame-stacking or normalization.

.. note::

	When using vectorized environments, the environments are automatically reset at the end of each episode.
	Thus, the observation returned for the i-th environment when ``done[i]`` is true will in fact be the first observation of the next episode, not the last observation of the episode that has just terminated.
	You can access the "real" final observation of the terminated episode—that is, the one that accompanied the ``done`` event provided by the underlying environment—using the ``terminal_observation`` keys in the info dicts returned by the ``VecEnv``.


.. warning::

  When defining a custom ``VecEnv`` (for instance, using gym3 ``ProcgenEnv``), you should provide ``terminal_observation`` keys in the info dicts returned by the ``VecEnv``
  (cf. note above).


.. warning::

    When using ``SubprocVecEnv``, users must wrap the code in an ``if __name__ == "__main__":`` if using the ``forkserver`` or ``spawn`` start method (default on Windows).
    On Linux, the default start method is ``fork`` which is not thread safe and can create deadlocks.

    For more information, see Python's `multiprocessing guidelines <https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods>`_.


VecEnv API vs Gym API
---------------------

For consistency across Stable-Baselines3 (SB3) versions and because of its special requirements and features,
SB3 VecEnv API is not the same as Gym API.
SB3 VecEnv API is actually close to Gym 0.21 API but differs to Gym 0.26+ API:

- the ``reset()`` method only returns the observation (``obs = vec_env.reset()``) and not a tuple, the info at reset are stored in ``vec_env.reset_infos``.

- only the initial call to ``vec_env.reset()`` is required, environments are reset automatically afterward (and ``reset_infos`` is updated automatically).

- the ``vec_env.step(actions)`` method expects an array as input
  (with a batch size corresponding to the number of environments) and returns a 4-tuple (and not a 5-tuple): ``obs, rewards, dones, infos`` instead of ``obs, reward, terminated, truncated, info``
  where ``dones = terminated or truncated`` (for each env).
  ``obs, rewards, dones`` are numpy arrays with shape ``(n_envs, shape_for_single_env)`` (so with a batch dimension).
  Additional information is passed via the ``infos`` value which is a list of dictionaries.

- at the end of an episode, ``infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated``
  tells the user if an episode was truncated or not:
  you should bootstrap if ``infos[env_idx]["TimeLimit.truncated"] is True`` (episode over due to a timeout/truncation)
  or ``dones[env_idx] is False`` (episode not finished).
  Note: compared to Gym 0.26+ ``infos[env_idx]["TimeLimit.truncated"]`` and ``terminated`` `are mutually exclusive <https://github.com/openai/gym/issues/3102>`_.
  The conversion from SB3 to Gym API is

  .. code-block:: python

    # done is True at the end of an episode
    # dones[env_idx] = terminated[env_idx] or truncated[env_idx]
    # In SB3, truncated and terminated are mutually exclusive
    # infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated
    # terminated[env_idx] tells you whether you should bootstrap or not:
    # when the episode has not ended or when the termination was a timeout/truncation
    terminated[env_idx] = dones[env_idx] and not infos[env_idx]["TimeLimit.truncated"]
    should_bootstrap[env_idx] = not terminated[env_idx]


- at the end of an episode, because the environment resets automatically,
  we provide ``infos[env_idx]["terminal_observation"]`` which contains the last observation
  of an episode (and can be used when bootstrapping, see note in the previous section)

- to overcome the current Gymnasium limitation (only one render mode allowed per env instance, see `issue #100 <https://github.com/Farama-Foundation/Gymnasium/issues/100>`_),
  we recommend using ``render_mode="rgb_array"`` since we can both have the image as a numpy array and display it with OpenCV.
  if no mode is passed or ``mode="rgb_array"`` is passed when calling ``vec_env.render`` then we use the default mode, otherwise, we use the OpenCV display.
  Note that if ``render_mode != "rgb_array"``, you can only call ``vec_env.render()`` (without argument or with ``mode=env.render_mode``).

- the ``reset()`` method doesn't take any parameter. If you want to seed the pseudo-random generator,
  you should call ``vec_env.seed(seed=seed)`` and ``obs = vec_env.reset()`` afterward.

- methods and attributes of the underlying Gym envs can be accessed, called and set using ``vec_env.get_attr("attribute_name")``,
  ``vec_env.env_method("method_name", args1, args2, kwargs1=kwargs1)`` and ``vec_env.set_attr("attribute_name", new_value)``.


Vectorized Environments Wrappers
--------------------------------

If you want to alter or augment a ``VecEnv`` without redefining it completely (e.g. stack multiple frames, monitor the ``VecEnv``, normalize the observation, ...), you can use ``VecEnvWrapper`` for that.
They are the vectorized equivalents (i.e., they act on multiple environments at the same time) of ``gym.Wrapper``.

You can find below an example for extracting one key from the observation:

.. code-block:: python

	import numpy as np

	from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper


	class VecExtractDictObs(VecEnvWrapper):
	    """
	    A vectorized wrapper for filtering a specific key from dictionary observations.
	    Similar to Gym's FilterObservation wrapper:
	        https://github.com/openai/gym/blob/master/gym/wrappers/filter_observation.py

	    :param venv: The vectorized environment
	    :param key: The key of the dictionary observation
	    """

	    def __init__(self, venv: VecEnv, key: str):
	        self.key = key
	        super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])

	    def reset(self) -> np.ndarray:
	        obs = self.venv.reset()
	        return obs[self.key]

	    def step_async(self, actions: np.ndarray) -> None:
	        self.venv.step_async(actions)

	    def step_wait(self) -> VecEnvStepReturn:
	        obs, reward, done, info = self.venv.step_wait()
	        return obs[self.key], reward, done, info

	env = DummyVecEnv([lambda: gym.make("FetchReach-v1")])
	# Wrap the VecEnv
	env = VecExtractDictObs(env, key="observation")


VecEnv
------

.. autoclass:: VecEnv
  :members:

DummyVecEnv
-----------

.. autoclass:: DummyVecEnv
  :members:

SubprocVecEnv
-------------

.. autoclass:: SubprocVecEnv
  :members:

Wrappers
--------

VecFrameStack
~~~~~~~~~~~~~

.. autoclass:: VecFrameStack
  :members:

StackedObservations
~~~~~~~~~~~~~~~~~~~

.. autoclass:: stable_baselines3.common.vec_env.stacked_observations.StackedObservations
  :members:

VecNormalize
~~~~~~~~~~~~

.. autoclass:: VecNormalize
  :members:


VecVideoRecorder
~~~~~~~~~~~~~~~~

.. autoclass:: VecVideoRecorder
  :members:


VecCheckNan
~~~~~~~~~~~~~~~~

.. autoclass:: VecCheckNan
  :members:


VecTransposeImage
~~~~~~~~~~~~~~~~~

.. autoclass:: VecTransposeImage
  :members:

VecMonitor
~~~~~~~~~~~~~~~~~

.. autoclass:: VecMonitor
  :members:

VecExtractDictObs
~~~~~~~~~~~~~~~~~

.. autoclass:: VecExtractDictObs
  :members:
