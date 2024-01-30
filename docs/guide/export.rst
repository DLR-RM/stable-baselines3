.. _export:


Exporting models
================

After training an agent, you may want to deploy/use it in another language
or framework, like `tensorflowjs <https://github.com/tensorflow/tfjs>`_.
Stable Baselines3 does not include tools to export models to other frameworks, but
this document aims to cover parts that are required for exporting along with
more detailed stories from users of Stable Baselines3.


Background
----------

In Stable Baselines3, the controller is stored inside policies which convert
observations into actions. Each learning algorithm (e.g. DQN, A2C, SAC)
contains a policy object which represents the currently learned behavior,
accessible via ``model.policy``.

Policies hold enough information to do the inference (i.e. predict actions),
so it is enough to export these policies (cf :ref:`examples <examples>`)
to do inference in another framework.

.. warning::
  When using CNN policies, the observation is normalized during pre-preprocessing.
  This pre-processing is done *inside* the policy (dividing by 255 to have values in [0, 1])


Export to ONNX
-----------------


If you are using PyTorch 2.0+ and ONNX Opset 14+, you can easily export SB3 policies using the following code:


.. warning::

  The following returns normalized actions and doesn't include the `post-processing <https://github.com/DLR-RM/stable-baselines3/blob/a9273f968eaf8c6e04302a07d803eebfca6e7e86/stable_baselines3/common/policies.py#L370-L377>`_ step that is done with continuous actions
  (clip or unscale the action to the correct space).


.. code-block:: python

  import torch as th
  from typing import Tuple

  from stable_baselines3 import PPO
  from stable_baselines3.common.policies import BasePolicy


  class OnnxableSB3Policy(th.nn.Module):
      def __init__(self, policy: BasePolicy):
          super().__init__()
          self.policy = policy

      def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
          # NOTE: Preprocessing is included, but postprocessing
          # (clipping/inscaling actions) is not,
          # If needed, you also need to transpose the images so that they are channel first
          # use deterministic=False if you want to export the stochastic policy
          # policy() returns `actions, values, log_prob` for PPO
          return self.policy(observation, deterministic=True)


  # Example: model = PPO("MlpPolicy", "Pendulum-v1")
  PPO("MlpPolicy", "Pendulum-v1").save("PathToTrainedModel")
  model = PPO.load("PathToTrainedModel.zip", device="cpu")

  onnx_policy = OnnxableSB3Policy(model.policy)

  observation_size = model.observation_space.shape
  dummy_input = th.randn(1, *observation_size)
  th.onnx.export(
      onnx_policy,
      dummy_input,
      "my_ppo_model.onnx",
      opset_version=17,
      input_names=["input"],
  )

  ##### Load and test with onnx

  import onnx
  import onnxruntime as ort
  import numpy as np

  onnx_path = "my_ppo_model.onnx"
  onnx_model = onnx.load(onnx_path)
  onnx.checker.check_model(onnx_model)

  observation = np.zeros((1, *observation_size)).astype(np.float32)
  ort_sess = ort.InferenceSession(onnx_path)
  actions, values, log_prob = ort_sess.run(None, {"input": observation})

  print(actions, values, log_prob)

  # Check that the predictions are the same
  with th.no_grad():
      print(model.policy(th.as_tensor(observation), deterministic=True))


For SAC the procedure is similar. The example shown only exports the actor network as the actor is sufficient to roll out the trained policies.

.. code-block:: python

  import torch as th

  from stable_baselines3 import SAC


  class OnnxablePolicy(th.nn.Module):
      def __init__(self, actor: th.nn.Module):
          super().__init__()
          self.actor = actor

      def forward(self, observation: th.Tensor) -> th.Tensor:
          # NOTE: You may have to postprocess (unnormalize) actions
          # to the correct bounds (see commented code below)
          return self.actor(observation, deterministic=True)


  # Example: model = SAC("MlpPolicy", "Pendulum-v1")
  SAC("MlpPolicy", "Pendulum-v1").save("PathToTrainedModel.zip")
  model = SAC.load("PathToTrainedModel.zip", device="cpu")
  onnxable_model = OnnxablePolicy(model.policy.actor)

  observation_size = model.observation_space.shape
  dummy_input = th.randn(1, *observation_size)
  th.onnx.export(
      onnxable_model,
      dummy_input,
      "my_sac_actor.onnx",
      opset_version=17,
      input_names=["input"],
  )

  ##### Load and test with onnx

  import onnxruntime as ort
  import numpy as np

  onnx_path = "my_sac_actor.onnx"

  observation = np.zeros((1, *observation_size)).astype(np.float32)
  ort_sess = ort.InferenceSession(onnx_path)
  scaled_action = ort_sess.run(None, {"input": observation})[0]

  print(scaled_action)

  # Post-process: rescale to correct space
  # Rescale the action from [-1, 1] to [low, high]
  # low, high = model.action_space.low, model.action_space.high
  # post_processed_action = low + (0.5 * (scaled_action + 1.0) * (high - low))

  # Check that the predictions are the same
  with th.no_grad():
      print(model.actor(th.as_tensor(observation), deterministic=True))


For more discussion around the topic, please refer to `GH#383 <https://github.com/DLR-RM/stable-baselines3/issues/383>`_ and `GH#1349 <https://github.com/DLR-RM/stable-baselines3/issues/1349>`_.



Trace/Export to C++
-------------------

You can use PyTorch JIT to trace and save a trained model that can be re-used in other applications
(for instance inference code written in C++).

There is a draft PR in the RL Zoo about C++ export: https://github.com/DLR-RM/rl-baselines3-zoo/pull/228

.. code-block:: python

  # See "ONNX export" for imports and OnnxablePolicy
  jit_path = "sac_traced.pt"

  # Trace and optimize the module
  traced_module = th.jit.trace(onnxable_model.eval(), dummy_input)
  frozen_module = th.jit.freeze(traced_module)
  frozen_module = th.jit.optimize_for_inference(frozen_module)
  th.jit.save(frozen_module, jit_path)

  ##### Load and test with torch

  import torch as th

  dummy_input = th.randn(1, *observation_size)
  loaded_module = th.jit.load(jit_path)
  action_jit = loaded_module(dummy_input)


Export to tensorflowjs / ONNX-JS
--------------------------------

TODO: contributors help is welcomed!
Probably a good starting point: https://github.com/elliotwaite/pytorch-to-javascript-with-onnx-js


Export to TFLite / Coral (Edge TPU)
-----------------------------------

Full example code: https://github.com/chunky/sb3_to_coral

Google created a chip called the "Coral" for deploying AI to the
edge. It's available in a variety of form factors, including USB (using
the Coral on a Raspberry Pi, with a SB3-developed model, was the original
motivation for the code example above).

The Coral chip is fast, with very low power consumption, but only has limited
on-device training abilities. More information is on the webpage here:
https://coral.ai.

To deploy to a Coral, one must work via TFLite, and quantize the
network to reflect the Coral's capabilities. The full chain to go from
SB3 to Coral is: SB3 (Torch) => ONNX => TensorFlow => TFLite => Coral.

The code linked above is a complete, minimal, example that:

1. Creates a model using SB3
2. Follows the path of exports all the way to TFLite and Google Coral
3. Demonstrates the forward pass for most exported variants

There are a number of pitfalls along the way to the complete conversion
that this example covers, including:

- Making the Gym's observation work with ONNX properly
- Quantising the TFLite model appropriately to align with Gym
  while still taking advantage of Coral
- Using OnnxablePolicy described as described in the above example


Manual export
-------------

You can also manually export required parameters (weights) and construct the
network in your desired framework.

You can access parameters of the model via agents'
:func:`get_parameters <stable_baselines3.common.base_class.BaseAlgorithm.get_parameters>` function.
As policies are also PyTorch modules, you can also access ``model.policy.state_dict()`` directly.
To find the architecture of the networks for each algorithm, best is to check the ``policies.py`` file located
in their respective folders.

.. note::

  In most cases, we recommend using PyTorch methods ``state_dict()`` and ``load_state_dict()`` from the policy,
  unless you need to access the optimizers' state dict too. In that case, you need to call ``get_parameters()``.
