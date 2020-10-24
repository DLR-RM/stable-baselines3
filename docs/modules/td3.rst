.. _td3:

.. automodule:: stable_baselines3.td3


TD3
===

`Twin Delayed DDPG (TD3) <https://spinningup.openai.com/en/latest/algorithms/td3.html>`_ Addressing Function Approximation Error in Actor-Critic Methods.

TD3 is a direct successor of DDPG and improves it using three major tricks: clipped double Q-Learning, delayed policy update and target policy smoothing.
We recommend reading `OpenAI Spinning guide on TD3 <https://spinningup.openai.com/en/latest/algorithms/td3.html>`_ to learn more about those.


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy


Notes
-----

- Original paper: https://arxiv.org/pdf/1802.09477.pdf
- OpenAI Spinning Guide for TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
- Original Implementation: https://github.com/sfujim/TD3

.. note::

    The default policies for TD3 differ a bit from others MlpPolicy: it uses ReLU instead of tanh activation,
    to match the original paper


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ❌
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ✔️
Box           ✔️       ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
============= ====== ===========


Example
-------

.. code-block:: python

  import gym
  import numpy as np

  from stable_baselines3 import TD3
  from stable_baselines3.td3.policies import MlpPolicy
  from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

  env = gym.make('Pendulum-v0')

  # The noise objects for TD3
  n_actions = env.action_space.shape[-1]
  action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

  model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1)
  model.learn(total_timesteps=10000, log_interval=10)
  model.save("td3_pendulum")
  env = model.get_env()

  del model # remove to demonstrate saving and loading

  model = TD3.load("td3_pendulum")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


Parameters
----------

.. autoclass:: TD3
  :members:
  :inherited-members:

.. _td3_policies:

TD3 Policies
-------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:

.. autoclass:: stable_baselines3.td3.policies.TD3Policy
  :members:
  :noindex:

.. autoclass:: CnnPolicy
  :members:
