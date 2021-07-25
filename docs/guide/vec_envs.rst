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

StackedDictObservations
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: stable_baselines3.common.vec_env.stacked_observations.StackedDictObservations
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
