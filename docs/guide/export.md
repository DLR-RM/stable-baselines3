(export)=

# Exporting models

After training an agent, you may want to deploy/use it in another language
or framework, like [tensorflowjs](https://github.com/tensorflow/tfjs).
Stable Baselines3 does not include tools to export models to other frameworks, but
this document aims to cover parts that are required for exporting along with
more detailed stories from users of Stable Baselines3.

## Background

In Stable Baselines3, the controller is stored inside policies which convert
observations into actions. Each learning algorithm (e.g. DQN, A2C, SAC)
contains a policy object which represents the currently learned behavior,
accessible via `model.policy`.

Policies hold enough information to do the inference (i.e. predict actions),
so it is enough to export these policies (cf {ref}`examples <examples>`)
to do inference in another framework.

:::{warning}
When using CNN policies, the observation is normalized during pre-preprocessing.
This pre-processing is done *inside* the policy (dividing by 255 to have values in [0, 1])
:::

## Export to ONNX

If you are using PyTorch 2.0+ and ONNX Opset 14+, you can easily export SB3 policies using the following code:

:::{warning}
The following returns normalized actions and doesn't include the [post-processing](https://github.com/DLR-RM/stable-baselines3/blob/a9273f968eaf8c6e04302a07d803eebfca6e7e86/stable_baselines3/common/policies.py#L370-L377) step that is done with continuous actions (clip or unscale the action to the correct space).
:::

```python
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
```

For exporting `MultiInputPolicy`, please have a look at [GH#1873](https://github.com/DLR-RM/stable-baselines3/issues/1873#issuecomment-2710776085).

For SAC the procedure is similar. The example shown only exports the actor network as the actor is sufficient to roll out the trained policies.

```python
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
```

For more discussion around the topic, please refer to [GH#383](https://github.com/DLR-RM/stable-baselines3/issues/383) and [GH#1349](https://github.com/DLR-RM/stable-baselines3/issues/1349).

## Trace/Export to C++

You can use PyTorch JIT to trace and save a trained model that can be reused in other applications
(for instance inference code written in C++).

There is a draft PR in the RL Zoo about C++ export: <https://github.com/DLR-RM/rl-baselines3-zoo/pull/228>

```python
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
```

## Export to ONNX-JS / ONNX Runtime Web

Official documentation: <https://onnxruntime.ai/docs/tutorials/web/build-web-app.html>

Full example code: <https://github.com/JonathanColetti/CarDodgingGym>

Demo: <https://jonathancoletti.github.io/CarDodgingGym>

The code linked above is a complete example (using car dodging environment) that:

1. Creates/Trains a PPO model
2. Exports the model to ONNX along with normalization stats in JSON
3. Runs in the browser with normalization using onnxruntime-web to achieve similar results

Below is a simple example with converting to ONNX then inferencing without postprocess in ONNX-JS

```python
import torch as th

from stable_baselines3 import SAC


class OnnxablePolicy(th.nn.Module):
    def __init__(self, actor: th.nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, observation: th.Tensor) -> th.Tensor:
        # NOTE: You may have to postprocess (unnormalize or renormalize)
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
```

```javascript
// Install using `npm install onnxruntime-web` (tested with version 1.19) or using cdn
import * as ort from 'onnxruntime-web';

async function runInference() {
  const session = await ort.InferenceSession.create('my_sac_actor.onnx');

  // The observation_size = 3 (for Pendulum-v1)
  const inputData = Float32Array.from([0.1, -0.2, 0.3]);

  const inputTensor = new ort.Tensor('float32', inputData, [1, 3]);

  const results = await session.run({ input: inputTensor });

  const outputName = session.outputNames[0];
  const action = results[outputName].data;

  console.log('Predicted action=', action);
}

runInference();
```

## Export to TensorFlow.js

:::{warning}
As of November 2025, [onnx2tf](https://github.com/PINTO0309/onnx2tf) does not support TensorFlow.js. Therefore, [tfjs-converter](https://github.com/tensorflow/tfjs-converter) is used instead. However, tfjs-converter is not currently maintained and requires older opsets and TensorFlow versions.
:::

In order for this to work, you have to do multiple conversions: SB3 => ONNX => TensorFlow => TensorFlow.js.

The opset version needs to be changed for the conversion (`opset_version=14` is currently required). Please refer to the code above for more stable usage with a higher opset.

The following is a simple example that showcases the full conversion + inference.

Please refer to the previous sections for the first step (SB3 => ONNX).
The main difference is that you need to specify `opset_version=14`.

```python
# Tested with python3.10
# Then install these dependencies in a fresh env
"""
pip install --use-deprecated=legacy-resolver tensorflow==2.13.0 keras==2.13.1 onnx==1.16.0 onnx-tf==1.9.0 tensorflow-probability==0.21.0 tensorflowjs==4.15.0 jax==0.4.26 jaxlib==0.4.26
"""
# Then run this codeblock
# If there are no errors (the folder is structure correctly) then
"""
# tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model tf_model tfjs_model
"""

# If you get an error exporting using `tensorflowjs_converter` then upgrade tensorflow
"""
pip install --upgrade tensorflow tensorflow-decision-forests tensorflowjs
"""
# And retry with and it should work (do not rerun this codeblock)
"""
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model tf_model tfjs_model
"""

import onnx
import onnx_tf.backend
import tensorflow as tf

ONNX_FILE_PATH = "my_sac_actor.onnx"
MODEL_PATH = "tf_model"

onnx_model = onnx.load(ONNX_FILE_PATH)
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

print('Converting ONNX to TF...')
tf_rep = onnx_tf.backend.prepare(onnx_model)
tf_rep.export_graph(MODEL_PATH)
# After this do not forget to use `tensorflowjs_converter`
```

```javascript
import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/+esm';
// Post processing not included
async function runInference() {
    const MODEL_URL = './tfjs_model/model.json';

    const model = await tf.loadGraphModel(MODEL_URL);

    // Observation_size is 3 for Pendulum-v1
    const inputData = [1.0, 0.0, 0.0];
    const inputTensor = tf.tensor2d([inputData], [1, 3]);

    const resultTensor = model.execute(inputTensor);

    const action = await resultTensor.data();

    console.log('Predicted action=', action);

    inputTensor.dispose();
    resultTensor.dispose();
}

runInference();
```

## Export to TFLite / Coral (Edge TPU)

Full example code: <https://github.com/chunky/sb3_to_coral>

Google created a chip called the "Coral" for deploying AI to the
edge. It's available in a variety of form factors, including USB (using
the Coral on a Raspberry Pi, with a SB3-developed model, was the original
motivation for the code example above).

The Coral chip is fast, with very low power consumption, but only has limited
on-device training abilities. More information is on the webpage here:
<https://coral.ai>.

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

## Manual export

You can also manually export required parameters (weights) and construct the
network in your desired framework.

You can access parameters of the model via agents'
{func}`get_parameters <stable_baselines3.common.base_class.BaseAlgorithm.get_parameters>` function.
As policies are also PyTorch modules, you can also access `model.policy.state_dict()` directly.
To find the architecture of the networks for each algorithm, best is to check the `policies.py` file located
in their respective folders.

:::{note}
In most cases, we recommend using PyTorch methods `state_dict()` and `load_state_dict()` from the policy,
unless you need to access the optimizers' state dict too. In that case, you need to call `get_parameters()`.
:::

## SBX (SB3 + Jax) Export

As an example of manual export, {ref}`Stable Baselines Jax (SBX) <sbx>` policies can be exported to ONNX
by using an intermediate PyTorch representation, as shown in the following example:

```python
import numpy as np
import sbx
import torch as th


class TorchPolicy(th.nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, act_dim: int):
        super().__init__()
        self.net = th.nn.Sequential(
            th.nn.Linear(obs_dim, hidden_dim),
            th.nn.Tanh(),
            th.nn.Linear(hidden_dim, hidden_dim),
            th.nn.Tanh(),
            th.nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


model = sbx.PPO("MlpPolicy", "Pendulum-v1")
# Also possible: load a trained model
# model = sbx.PPO.load("PathToTrainedModel.zip")

params = model.policy.actor_state.params["params"]
# For debug:
print("=== SBX params ===")
for key, value in params.items():
    if isinstance(value, dict):
        for name, val in value.items():
            print(f"{key}.{name}: {val.shape}", end=" ")
    else:
        print(f"{key}: {value.shape}", end=" ")
print("\n" + "=" * 20 + "\n")

obs_dim = model.observation_space.shape
act_dim = model.action_space.shape

# Number of units in the hidden layers (assume a network architecture like [64, 64])
hidden_dim = params["Dense_0"]["kernel"].shape[1]

# map params to torch state_dict keys
num_layers = len([k for k in params.keys() if k.startswith("Dense_")])
state_dict = {}
for i in range(num_layers):
    layer_name = f"Dense_{i}"
    state_dict[f"net.{i * 2}.bias"] = th.from_numpy(np.array(params[layer_name]["bias"]))
    state_dict[f"net.{i * 2}.weight"] = th.from_numpy(np.array(params[layer_name]["kernel"].T))

torch_policy = TorchPolicy(obs_dim[0], hidden_dim, act_dim[0])
print("=== Torch params ===")
print(" ".join(f"{key}:{tuple(value.shape)}" for key, value in torch_policy.named_parameters()))
print("=" * 20 + "\n")

torch_policy.load_state_dict(state_dict)
torch_policy.eval()

dummy_input = th.zeros((1, *obs_dim))
# Use normal Torch export
th.onnx.export(
    torch_policy,
    (dummy_input,),
    "my_ppo_actor.onnx",
    opset_version=18,
    input_names=["input"],
    output_names=["action"],
)


##### Load and test with onnx

import onnxruntime as ort

onnx_path = "my_ppo_actor.onnx"
ort_sess = ort.InferenceSession(onnx_path)

observation = np.random.random((1, *obs_dim)).astype(np.float32)
action = ort_sess.run(None, {"input": observation})[0]

print(action)
sbx_action, _ = model.predict(observation, deterministic=True)
with th.no_grad():
    torch_action = torch_policy(th.as_tensor(observation))

# Check that the predictions are the same
assert np.allclose(sbx_action, action)
assert np.allclose(sbx_action, torch_action.numpy())
```
