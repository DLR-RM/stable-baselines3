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

As of June 2021, ONNX format  `doesn't support <https://github.com/onnx/onnx/issues/3033>`_ exporting models that use the ``broadcast_tensors`` functionality of pytorch. So in order to export the trained stable-baseline3 models in the ONNX format, we need to first remove the layers that use broadcasting. This can be done by creating a class that removes the unsupported layers.

The following examples are for ``MlpPolicy`` only, and are general examples. Note that you have to preprocess the observation the same way stable-baselines3 agent does (see ``common.preprocessing.preprocess_obs``).

For PPO, assuming a shared feature extractor.

.. warning::

  The following example is for continuous actions only.
  When using discrete or binary actions, you must do some `post-processing <https://github.com/DLR-RM/stable-baselines3/blob/f3a35aa786ee41ffff599b99fa1607c067e89074/stable_baselines3/common/policies.py#L621-L637>`_
  to obtain the action (e.g., convert action logits to action).


.. code-block:: python

  import torch as th

  from stable_baselines3 import PPO


  class OnnxablePolicy(th.nn.Module):
      def __init__(self, extractor, action_net, value_net):
          super().__init__()
          self.extractor = extractor
          self.action_net = action_net
          self.value_net = value_net

      def forward(self, observation):
          # NOTE: You may have to process (normalize) observation in the correct
          #       way before using this. See `common.preprocessing.preprocess_obs`
          action_hidden, value_hidden = self.extractor(observation)
          return self.action_net(action_hidden), self.value_net(value_hidden)


  # Example: model = PPO("MlpPolicy", "Pendulum-v1")
  model = PPO.load("PathToTrainedModel.zip", device="cpu")
  onnxable_model = OnnxablePolicy(
      model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net
  )

  observation_size = model.observation_space.shape
  dummy_input = th.randn(1, *observation_size)
  th.onnx.export(
      onnxable_model,
      dummy_input,
      "my_ppo_model.onnx",
      opset_version=9,
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
  action, value = ort_sess.run(None, {"input": observation})


For SAC the procedure is similar. The example shown only exports the actor network as the actor is sufficient to roll out the trained policies.

.. code-block:: python

  import torch as th

  from stable_baselines3 import SAC


  class OnnxablePolicy(th.nn.Module):
      def __init__(self, actor: th.nn.Module):
          super().__init__()
          # Removing the flatten layer because it can't be onnxed
          self.actor = th.nn.Sequential(
              actor.latent_pi,
              actor.mu,
              # For gSDE
              # th.nn.Hardtanh(min_val=-actor.clip_mean, max_val=actor.clip_mean),
              # Squash the output
              th.nn.Tanh(),
          )

      def forward(self, observation: th.Tensor) -> th.Tensor:
          # NOTE: You may have to process (normalize) observation in the correct
          #       way before using this. See `common.preprocessing.preprocess_obs`
          return self.actor(observation)


  # Example: model = SAC("MlpPolicy", "Pendulum-v1")
  model = SAC.load("PathToTrainedModel.zip", device="cpu")
  onnxable_model = OnnxablePolicy(model.policy.actor)

  observation_size = model.observation_space.shape
  dummy_input = th.randn(1, *observation_size)
  th.onnx.export(
      onnxable_model,
      dummy_input,
      "my_sac_actor.onnx",
      opset_version=9,
      input_names=["input"],
  )

  ##### Load and test with onnx

  import onnxruntime as ort
  import numpy as np

  onnx_path = "my_sac_actor.onnx"

  observation = np.zeros((1, *observation_size)).astype(np.float32)
  ort_sess = ort.InferenceSession(onnx_path)
  action = ort_sess.run(None, {"input": observation})


For more discussion around the topic refer to this `issue. <https://github.com/DLR-RM/stable-baselines3/issues/383>`_

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
the Coral on a Rasbperry pi, with a SB3-developed model, was the original
motivation for the code example above).

The Coral chip is fast, with very low power consumption, but only has limited
on-device training abilities. More information is on the webpage here:
https://coral.ai.

To deploy to a Coral, one must work via TFLite, and quantise the
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
