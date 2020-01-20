.. _cem_rl:

.. automodule:: torchy_baselines.cem_rl


CEM RL
======

Combining cross-entropy method (CEM) and Twin Delayed Deep Deterministic policy gradient (TD3).


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy


Notes
-----

- Original paper: https://arxiv.org/abs/1810.01222 and https://openreview.net/forum?id=BkeU5j0ctQ
- Original Implementation: https://github.com/apourchot/CEM-RL


.. note::

	CEM RL is currently implemented for TD3


.. note::

    The default policies for CEM RL differ a bit from others MlpPolicy: it uses ReLU instead of tanh activation,
    to match the original paper


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ❌
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ❌
Box           ✔️       ✔️
MultiDiscrete ❌      ❌
MultiBinary   ❌      ❌
============= ====== ===========


Example
-------

.. code-block:: python

  import numpy as np

  from torchy_baselines import CEMRL
  from torchy_baselines.td3.policies import MlpPolicy

  # n_grad = 0 corresponds to CEM (in fact CMA-ES without history)
  model = CEMRL(MlpPolicy, 'Pendulum-v0', pop_size=10, n_grad=5, verbose=1)
  model.learn(total_timesteps=50000, log_interval=10)
  model.save("td3_pendulum")
  env = model.get_env()

  del model # remove to demonstrate saving and loading

  model = CEMRL.load("td3_pendulum")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()

Parameters
----------

.. autoclass:: CEMRL
  :members:
  :inherited-members:

.. _cemrl_policies:

CEM RL Policies
---------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:
