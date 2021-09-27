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

The following examples are for ``MlpPolicy`` only, and are general examples. Note that you have to preprocess the observation the same way stable-baselines3 agent does (see ``common.preprocessing.preprocess_obs``)

For PPO, assuming a shared feature extactor.

.. warning::

  The following example is for continuous actions only.
  When using discrete or binary actions, you must do some `post-processing <https://github.com/DLR-RM/stable-baselines3/blob/f3a35aa786ee41ffff599b99fa1607c067e89074/stable_baselines3/common/policies.py#L621-L637>`_
  to obtain the action (e.g., convert action logits to action).


.. code-block:: python

  from stable_baselines3 import PPO
  import torch

  class OnnxablePolicy(torch.nn.Module):
    def __init__(self, extractor, action_net, value_net):
        super(OnnxablePolicy, self).__init__()
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net

    def forward(self, observation):
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        action_hidden, value_hidden = self.extractor(observation)
        return self.action_net(action_hidden), self.value_net(value_hidden)

  # Example: model = PPO("MlpPolicy", "Pendulum-v0")
  model = PPO.load("PathToTrainedModel.zip")
  model.policy.to("cpu")
  onnxable_model = OnnxablePolicy(model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net)

  dummy_input = torch.randn(1, observation_size)
  torch.onnx.export(onnxable_model, dummy_input, "my_ppo_model.onnx", opset_version=9)

  ##### Load and test with onnx

  import onnx
  import onnxruntime as ort
  import numpy as np

  onnx_model = onnx.load(onnx_path)
  onnx.checker.check_model(onnx_model)

  observation = np.zeros((1, observation_size)).astype(np.float32)
  ort_sess = ort.InferenceSession(onnx_path)
  action, value = ort_sess.run(None, {'input.1': observation})


For SAC the procedure is similar. The example shown only exports the actor network as the actor is sufficient to roll out the trained policies.

.. code-block:: python

  from stable_baselines3 import SAC
  import torch

  class OnnxablePolicy(torch.nn.Module):
    def __init__(self,  actor):
        super(OnnxablePolicy, self).__init__()

        # Removing the flatten layer because it can't be onnxed
        self.actor = torch.nn.Sequential(actor.latent_pi, actor.mu)

    def forward(self, observation):
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        return self.actor(observation)

  model = SAC.load("PathToTrainedModel.zip")
  onnxable_model = OnnxablePolicy(model.policy.actor)

  dummy_input = torch.randn(1, observation_size)
  onnxable_model.policy.to("cpu")
  torch.onnx.export(onnxable_model, dummy_input, "my_sac_actor.onnx", opset_version=9)


For more discussion around the topic refer to this `issue. <https://github.com/DLR-RM/stable-baselines3/issues/383>`_

Export to C++
-----------------

(using PyTorch JIT)
TODO: help is welcomed!


Export to tensorflowjs / ONNX-JS
--------------------------------

TODO: contributors help is welcomed!
Probably a good starting point: https://github.com/elliotwaite/pytorch-to-javascript-with-onnx-js



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
