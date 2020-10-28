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

TODO: help is welcomed!


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
