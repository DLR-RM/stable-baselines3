.. _custom_policy:

Custom Policy Network
---------------------

Stable Baselines3 provides policy networks for images (CnnPolicies)
and other type of input features (MlpPolicies).

One way of customising the policy network architecture is to pass arguments when creating the model,
using ``policy_kwargs`` parameter:

.. code-block:: python

  import gym
  import torch as th

  from stable_baselines3 import PPO

  # Custom MLP policy of two layers of size 32 each with tanh activation function
  policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[32, 32])
  # Create the agent
  model = PPO("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)
  # Retrieve the environment
  env = model.get_env()
  # Train the agent
  model.learn(total_timesteps=100000)
  # Save the agent
  model.save("ppo-cartpole")

  del model
  # the policy_kwargs are automatically loaded
  model = PPO.load("ppo-cartpole")


You can also easily define a custom architecture for the policy (or value) network:

.. note::

    Defining a custom policy class is equivalent to passing ``policy_kwargs``.
    However, it lets you name the policy and so makes usually the code clearer.
    ``policy_kwargs`` should be rather used when doing hyperparameter search.



The ``net_arch`` parameter of ``A2C`` and ``PPO`` policies allows to specify the amount and size of the hidden layers and how many
of them are shared between the policy network and the value network. It is assumed to be a list with the following
structure:

1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
   If the number of ints is zero, there will be no shared layers.
2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
   It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
   If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

In short: ``[<shared layers>, dict(vf=[<non-shared value network layers>], pi=[<non-shared policy network layers>])]``.

Examples
~~~~~~~~

Two shared layers of size 128: ``net_arch=[128, 128]``


.. code-block:: none

                  obs
                   |
                 <128>
                   |
                 <128>
           /               \
        action            value


Value network deeper than policy network, first layer shared: ``net_arch=[128, dict(vf=[256, 256])]``

.. code-block:: none

                  obs
                   |
                 <128>
           /               \
        action             <256>
                             |
                           <256>
                             |
                           value


Initially shared then diverging: ``[128, dict(vf=[256], pi=[16])]``

.. code-block:: none

                  obs
                   |
                 <128>
           /               \
         <16>             <256>
           |                |
        action            value



If your task requires even more granular control over the policy architecture, you can redefine the policy directly.

**TODO**
