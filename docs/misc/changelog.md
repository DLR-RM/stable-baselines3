(changelog)=

# Changelog

## Release 2.8.0a3 (WIP)

### Breaking Changes:

- Removed support for Python 3.9, please upgrade to Python >= 3.10
- Set `strict=True` for every call to `zip(...)`

### New Features:

- Added official support for Python 3.13

### Bug Fixes:

- Fixed saving and loading of Torch compiled models (using `th.compile()`) by updating `get_parameters()`
- Added a warning to env-checker if a multidiscrete space has multi-dimensional array (@unexploredtest)
- Fixed `pandas.concat` futurewarnings occuring when dataframes are empty by removing empty frames from the list before concatenating

### [SB3-Contrib]

### [RL Zoo]

### [SBX] (SB3 + Jax)

### Deprecations:

- `zip_strict()` is not needed anymore since Python 3.10, please use `zip(..., strict=True)` instead

### Others:

- Updated to Python 3.10+ annotations
- Removed some unused variables (@unexploredtest)
- Improved type hints for distributions
- Simplified zip file loading by removing Python 3.6 workaround and enabling `weights_only=True` (PyTorch 2.x)
- Sped up saving/loading tests
- Updated black from v25 to v26
- Updated monitor test to check handling of empty monitor files

### Documentation:

- Added a note on MultiDiscrete spaces with multi-dimensional arrays and a wrapper to fix the issue (@unexploredtest)
- Added an example of manual export of SBX (SB3 + Jax) model to ONNX (@m-abr)
- Switched to mardown documentation (using MyST parser)

## Release 2.7.1 (2025-12-05)

:::{warning}
Stable-Baselines3 (SB3) v2.7.1 will be the last one supporting Python 3.9 (end of life in October 2025).
We highly recommended you to upgrade to Python >= 3.10.
:::

### Breaking Changes:

### New Features:

- `RolloutBuffer` and `DictRolloutBuffer` now uses the actual observation / action space `dtype` (instead of float32), this should save memory (@Trenza1ore)

### Bug Fixes:

- Fixed env checker to properly handle `Sequence` observation spaces when nested inside composite spaces (`Dict`, `Tuple`, `OneOf`) (@copilot)
- Update env checker to warn users when using Graph space (@dhruvmalik007).
- Fixed memory leak in `VecVideoRecorder` where `recorded_frames` stayed in memory due to reference in the moviepy clip (@copilot)
- Remove double space in `StopTrainingOnRewardThreshold` callback message (@sea-bass)

### [SB3-Contrib]

- Fixed tensorboard log name for `MaskablePPO`

### [SBX] (SB3 + Jax)

- Added `CnnPolicy` to PPO

### Documentation:

- Added plotting documentation and examples
- Added documentation clarifying gSDE (Generalized State-Dependent Exploration) inference behavior for PPO, SAC, and A2C algorithms
- Documented Atari wrapper reset behavior where `env.reset()` may perform a no-op step instead of truly resetting when `terminal_on_life_loss=True` (default), and how to avoid this behavior by setting `terminal_on_life_loss=False`
- Clarified comment in `_sample_action()` method to better explain action scaling behavior for off-policy algorithms (@copilot)
- Added sb3-plus to projects page
- Added example usage of ONNX JS
- Updated link to paper of community project DeepNetSlice (@AlexPasqua)
- Added example usage of Tensorflow JS
- Included exact versions in ONNX JS and example project
- Made step 2 (`pip install`) of `CONTRIBUTING.md` more robust

## Release 2.7.0 (2025-07-25)

**n-step returns for all off-policy algorithms**

### Breaking Changes:

### New Features:

- Added support for n-step returns for off-policy algorithms via the `n_steps` parameter
- Added `NStepReplayBuffer` that allows to compute n-step returns without additional memory requirement (and without for loops)
- Added Gymnasium v1.2 support

### Bug Fixes:

- Fixed docker GPU image (PyTorch GPU was not installed)
- Fixed segmentation faults caused by non-portable schedules during model loading (@akanto)

### [SB3-Contrib]

- Added support for n-step returns for off-policy algorithms via the `n_steps` parameter
- Use the `FloatSchedule` and `LinearSchedule` classes instead of lambdas in the ARS, PPO, and QRDQN implementations to improve model portability across different operating systems

### [RL Zoo]

- `linear_schedule` now returns a `SimpleLinearSchedule` object for better portability
- Renamed `LunarLander-v2` to `LunarLander-v3` in hyperparameters
- Renamed `CarRacing-v2` to `CarRacing-v3` in hyperparameters
- Docker GPU images are now working again
- Use `ConstantSchedule`, and `SimpleLinearSchedule` instead of `constant_fn` and `linear_schedule`
- Fixed `CarRacing-v3` hyperparameters for newer Gymnasium version

### [SBX] (SB3 + Jax)

- Added support for n-step returns for off-policy algorithms via the `n_steps` parameter
- Added KL Adaptive LR for PPO and LR schedule for SAC/TQC

### Deprecations:

- `get_schedule_fn()`, `get_linear_fn()`, `constant_fn()` are deprecated, please use `FloatSchedule()`, `LinearSchedule()`, `ConstantSchedule()` instead

### Others:

### Documentation:

- Clarify `evaluate_policy` documentation
- Added doc about training exceeding the `total_timesteps` parameter
- Updated `LunarLander` and `LunarLanderContinuous` environment versions to v3 (@j0m0k0)
- Added sb3-extra-buffers to the project page (@Trenza1ore)

## Release 2.6.0 (2025-03-24)

**New \`\`LogEveryNTimesteps\`\` callback and \`\`has_attr\`\` method, refactored hyperparameter optimization**

### Breaking Changes:

### New Features:

- Added `has_attr` method for `VecEnv` to check if an attribute exists
- Added `LogEveryNTimesteps` callback to dump logs every N timesteps (note: you need to pass `log_interval=None` to avoid any interference)
- Added Gymnasium v1.1 support

### Bug Fixes:

- `SubProcVecEnv` will now exit gracefully (without big traceback) when using `KeyboardInterrupt`

### [SB3-Contrib]

- Renamed `_dump_logs()` to `dump_logs()`
- Fixed issues with `SubprocVecEnv` and `MaskablePPO` by using `vec_env.has_attr()` (pickling issues, mask function not present)

### [RL Zoo]

- Refactored hyperparameter optimization. The Optuna [Journal storage backend](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.JournalStorage.html) is now supported (recommended default) and you can easily load tuned hyperparameter via the new `--trial-id` argument of `train.py`.
- Save the exact command line used to launch a training
- Added support for special vectorized env (e.g. Brax, IsaacSim) by allowing to override the `VecEnv` class use to instantiate the env in the `ExperimentManager`
- Allow to disable auto-logging by passing `--log-interval -2` (useful when logging things manually)
- Added Gymnasium v1.1 support
- Fixed use of old HF api in `get_hf_trained_models()`

### [SBX] (SB3 + Jax)

- Updated PPO to support `net_arch`, and additional fixes
- Fixed entropy coeff wrongly logged for SAC and derivatives.
- Fixed PPO `predict()` for env that were not normalized (action spaces with limits != [-1, 1])
- PPO now logs the standard deviation

### Deprecations:

- `algo._dump_logs()` is deprecated in favor of `algo.dump_logs()` and will be removed in SB3 v2.7.0

### Others:

- Updated black from v24 to v25
- Improved error messages when checking Box space equality (loading `VecNormalize`)
- Updated test to reflect how `set_wrapper_attr` should be used now

### Documentation:

- Clarify the use of Gym wrappers with `make_vec_env` in the section on Vectorized Environments (@pstahlhofen)
- Updated callback doc for `EveryNTimesteps`
- Added doc on how to set env attributes via `VecEnv` calls
- Added ONNX export example for `MultiInputPolicy` (@darkopetrovic)

## Release 2.5.0 (2025-01-27)

**New algorithm: SimBa in SBX, NumPy 2.0 support**

### Breaking Changes:

- Increased minimum required version of PyTorch to 2.3.0
- Removed support for Python 3.8

### New Features:

- Added support for NumPy v2.0: `VecNormalize` now cast normalized rewards to float32, updated bit flipping env to avoid overflow issues too
- Added official support for Python 3.12

### [SBX] (SB3 + Jax)

- Added SimBa Policy: Simplicity Bias for Scaling Up Parameters in DRL
- Added support for parameter resets

### Others:

- Updated Dockerfile

### Documentation:

- Added Decisions and Dragons to resources. (@jmacglashan)
- Updated PyBullet example, now compatible with Gymnasium
- Added link to policies for `policy_kwargs` parameter (@kplers)
- Add FootstepNet Envs to the project page (@cgaspard3333)
- Added FRASA to the project page (@MarcDcls)
- Fixed atari example (@chrisgao99)
- Add a note about `Discrete` action spaces with `start!=0`
- Update doc for massively parallel simulators (Isaac Lab, Brax, ...)
- Add dm_control example

## Release 2.4.1 (2024-12-20)

### Bug Fixes:

- Fixed a bug introduced in v2.4.0 where the `VecVideoRecorder` would override videos

## Release 2.4.0 (2024-11-18)

**New algorithm: CrossQ in SB3 Contrib, Gymnasium v1.0 support**

:::{note}
DQN (and QR-DQN) models saved with SB3 < 2.4.0 will show a warning about
truncation of optimizer state when loaded with SB3 >= 2.4.0.
To suppress the warning, simply save the model again.
You can find more info in [PR #1963](https://github.com/DLR-RM/stable-baselines3/pull/1963)
:::

:::{warning}
Stable-Baselines3 (SB3) v2.4.0 will be the last one supporting Python 3.8 (end of life in October 2024)
and PyTorch < 2.3.
We highly recommended you to upgrade to Python >= 3.9 and PyTorch >= 2.3 (compatible with NumPy v2).
:::

### Breaking Changes:

- Increased minimum required version of Gymnasium to 0.29.1

### New Features:

- Added support for `pre_linear_modules` and `post_linear_modules` in `create_mlp` (useful for adding normalization layers, like in DroQ or CrossQ)
- Enabled np.ndarray logging for TensorBoardOutputFormat as histogram (see GH#1634) (@iwishwasaneagle)
- Updated env checker to warn users when using multi-dim array to define `MultiDiscrete` spaces
- Added support for Gymnasium v1.0

### Bug Fixes:

- Fixed memory leak when loading learner from storage, `set_parameters()` does not try to load the object data anymore
  and only loads the PyTorch parameters (@peteole)
- Cast type in compute gae method to avoid error when using torch compile (@amjames)
- `CallbackList` now sets the `.parent` attribute of child callbacks to its own `.parent`. (will-maclean)
- Fixed error when loading a model that has `net_arch` manually set to `None` (@jak3122)
- Set requirement numpy\<2.0 until PyTorch is compatible (<https://github.com/pytorch/pytorch/issues/107302>)
- Updated DQN optimizer input to only include q_network parameters, removing the target_q_network ones (@corentinlger)
- Fixed `test_buffers.py::test_device` which was not actually checking the device of tensors (@rhaps0dy)

### [SB3-Contrib]

- Added `CrossQ` algorithm, from "Batch Normalization in Deep Reinforcement Learning" paper (@danielpalen)
- Added `BatchRenorm` PyTorch layer used in `CrossQ` (@danielpalen)
- Updated QR-DQN optimizer input to only include quantile_net parameters (@corentinlger)
- Fixed loading QRDQN changes `target_update_interval` (@jak3122)

### [RL Zoo]

- Updated defaults hyperparameters for TQC/SAC for Swimmer-v4 (decrease gamma for more consistent results)

### [SBX] (SB3 + Jax)

- Added CNN support for DQN
- Bug fix for SAC and related algorithms, optimize log of ent coeff to be consistent with SB3

### Deprecations:

### Others:

- Fixed various typos (@cschindlbeck)
- Remove unnecessary SDE noise resampling in PPO update (@brn-dev)
- Updated PyTorch version on CI to 2.3.1
- Added a warning to recommend using CPU with on policy algorithms (A2C/PPO) and `MlpPolicy`
- Switched to uv to download packages faster on GitHub CI
- Updated dependencies for read the doc
- Removed unnecessary `copy_obs_dict` method for `SubprocVecEnv`, remove the use of ordered dict and rename `flatten_obs` to `stack_obs`

### Documentation:

- Updated PPO doc to recommend using CPU with `MlpPolicy`
- Clarified documentation about planned features and citing software
- Added a note about the fact we are optimizing log of ent coeff for SAC

## Release 2.3.2 (2024-04-27)

### Bug Fixes:

- Reverted `torch.load()` to be called `weights_only=False` as it caused loading issue with old version of PyTorch.

### Documentation:

- Added ER-MRL to the project page (@corentinlger)
- Updated Tensorboard Logging Videos documentation (@NickLucche)

## Release 2.3.1 (2024-04-22)

### Bug Fixes:

- Cast return value of learning rate schedule to float, to avoid issue when loading model because of `weights_only=True` (@markscsmith)

### Documentation:

- Updated SBX documentation (CrossQ and deprecated DroQ)
- Updated RL Tips and Tricks section

## Release 2.3.0 (2024-03-31)

**New defaults hyperparameters for DDPG, TD3 and DQN**

:::{warning}
Because of `weights_only=True`, this release breaks loading of policies when using PyTorch 1.13.
Please upgrade to PyTorch >= 2.0 or upgrade SB3 version (we reverted the change in SB3 2.3.2)
:::

### Breaking Changes:

- The defaults hyperparameters of `TD3` and `DDPG` have been changed to be more consistent with `SAC`

```python
# SB3 < 2.3.0 default hyperparameters
# model = TD3("MlpPolicy", env, train_freq=(1, "episode"), gradient_steps=-1, batch_size=100)
# SB3 >= 2.3.0:
model = TD3("MlpPolicy", env, train_freq=1, gradient_steps=1, batch_size=256)
```

:::{note}
Two inconsistencies remain: the default network architecture for `TD3/DDPG` is `[400, 300]` instead of `[256, 256]` for SAC (for backward compatibility reasons, see [report on the influence of the network size](https://wandb.ai/openrlbenchmark/sbx/reports/SBX-TD3-Influence-of-policy-net--Vmlldzo2NDg1Mzk3)) and the default learning rate is 1e-3 instead of 3e-4 for SAC (for performance reasons, see [W&B report on the influence of the lr](https://wandb.ai/openrlbenchmark/sbx/reports/SBX-TD3-RL-Zoo-v2-3-0a0-vs-SB3-TD3-RL-Zoo-2-2-1---Vmlldzo2MjUyNTQx))
:::

- The default `learning_starts` parameter of `DQN` have been changed to be consistent with the other offpolicy algorithms

```python
# SB3 < 2.3.0 default hyperparameters, 50_000 corresponded to Atari defaults hyperparameters
# model = DQN("MlpPolicy", env, learning_starts=50_000)
# SB3 >= 2.3.0:
model = DQN("MlpPolicy", env, learning_starts=100)
```

- For safety, `torch.load()` is now called with `weights_only=True` when loading torch tensors,
  policy `load()` still uses `weights_only=False` as gymnasium imports are required for it to work
- When using `huggingface_sb3`, you will now need to set `TRUST_REMOTE_CODE=True` when downloading models from the hub, as `pickle.load` is not safe.

### New Features:

- Log success rate `rollout/success_rate` when available for on policy algorithms (@corentinlger)

### Bug Fixes:

- Fixed `monitor_wrapper` argument that was not passed to the parent class, and dones argument that wasn't passed to `_update_into_buffer` (@corentinlger)

### [SB3-Contrib]

- Added `rollout_buffer_class` and `rollout_buffer_kwargs` arguments to MaskablePPO
- Fixed `train_freq` type annotation for tqc and qrdqn (@Armandpl)
- Fixed `sb3_contrib/common/maskable/*.py` type annotations
- Fixed `sb3_contrib/ppo_mask/ppo_mask.py` type annotations
- Fixed `sb3_contrib/common/vec_env/async_eval.py` type annotations
- Add some additional notes about `MaskablePPO` (evaluation and multi-process) (@icheered)

### [RL Zoo]

- Updated defaults hyperparameters for TD3/DDPG to be more consistent with SAC
- Upgraded MuJoCo envs hyperparameters to v4 (pre-trained agents need to be updated)
- Added test dependencies to `setup.py` (@power-edge)
- Simplify dependencies of `requirements.txt` (remove duplicates from `setup.py`)

### [SBX] (SB3 + Jax)

- Added support for `MultiDiscrete` and `MultiBinary` action spaces to PPO
- Added support for large values for gradient_steps to SAC, TD3, and TQC
- Fix `train()` signature and update type hints
- Fix replay buffer device at load time
- Added flatten layer
- Added `CrossQ`

### Deprecations:

### Others:

- Updated black from v23 to v24
- Updated ruff to >= v0.3.1
- Updated env checker for (multi)discrete spaces with non-zero start.

### Documentation:

- Added a paragraph on modifying vectorized environment parameters via setters (@fracapuano)
- Updated callback code example
- Updated export to ONNX documentation, it is now much simpler to export SB3 models with newer ONNX Opset!
- Added video link to "Practical Tips for Reliable Reinforcement Learning" video
- Added `render_mode="human"` in the README example (@marekm4)
- Fixed docstring signature for sum_independent_dims (@stagoverflow)
- Updated docstring description for `log_interval` in the base class (@rushitnshah).

## Release 2.2.1 (2023-11-17)

**Support for options at reset, bug fixes and better error messages**

:::{note}
SB3 v2.2.0 was yanked after a breaking change was found in [GH#1751](https://github.com/DLR-RM/stable-baselines3/issues/1751).
Please use SB3 v2.2.1 and not v2.2.0.
:::

### Breaking Changes:

- Switched to `ruff` for sorting imports (isort is no longer needed), black and ruff version now require a minimum version
- Dropped `x is False` in favor of `not x`, which means that callbacks that wrongly returned None (instead of a boolean) will cause the training to stop (@iwishiwasaneagle)

### New Features:

- Improved error message of the `env_checker` for env wrongly detected as GoalEnv (`compute_reward()` is defined)
- Improved error message when mixing Gym API with VecEnv API (see GH#1694)
- Add support for setting `options` at reset with VecEnv via the `set_options()` method. Same as seeds logic, options are reset at the end of an episode (@ReHoss)
- Added `rollout_buffer_class` and `rollout_buffer_kwargs` arguments to on-policy algorithms (A2C and PPO)

### Bug Fixes:

- Prevents using squash_output and not use_sde in ActorCritcPolicy (@PatrickHelm)
- Performs unscaling of actions in collect_rollout in OnPolicyAlgorithm (@PatrickHelm)
- Moves VectorizedActionNoise into `_setup_learn()` in OffPolicyAlgorithm (@PatrickHelm)
- Prevents out of bound error on Windows if no seed is passed (@PatrickHelm)
- Calls `callback.update_locals()` before `callback.on_rollout_end()` in OnPolicyAlgorithm (@PatrickHelm)
- Fixed replay buffer device after loading in OffPolicyAlgorithm (@PatrickHelm)
- Fixed `render_mode` which was not properly loaded when using `VecNormalize.load()`
- Fixed success reward dtype in `SimpleMultiObsEnv` (@NixGD)
- Fixed check_env for Sequence observation space (@corentinlger)
- Prevents instantiating BitFlippingEnv with conflicting observation spaces (@kylesayrs)
- Fixed ResourceWarning when loading and saving models (files were not closed), please note that only path are closed automatically,
  the behavior stay the same for tempfiles (they need to be closed manually),
  the behavior is now consistent when loading/saving replay buffer

### [SB3-Contrib]

- Added `set_options` for `AsyncEval`
- Added `rollout_buffer_class` and `rollout_buffer_kwargs` arguments to TRPO

### [RL Zoo]

- Removed `gym` dependency, the package is still required for some pretrained agents.
- Added `--eval-env-kwargs` to `train.py` (@Quentin18)
- Added `ppo_lstm` to hyperparams_opt.py (@technocrat13)
- Upgraded to `pybullet_envs_gymnasium>=0.4.0`
- Removed old hacks (for instance limiting offpolicy algorithms to one env at test time)
- Updated docker image, removed support for X server
- Replaced deprecated `optuna.suggest_uniform(...)` by `optuna.suggest_float(..., low=..., high=...)`

### [SBX] (SB3 + Jax)

- Added `DDPG` and `TD3` algorithms

### Deprecations:

### Others:

- Fixed `stable_baselines3/common/callbacks.py` type hints
- Fixed `stable_baselines3/common/utils.py` type hints
- Fixed `stable_baselines3/common/vec_envs/vec_transpose.py` type hints
- Fixed `stable_baselines3/common/vec_env/vec_video_recorder.py` type hints
- Fixed `stable_baselines3/common/save_util.py` type hints
- Updated docker images to Ubuntu Jammy using micromamba 1.5
- Fixed `stable_baselines3/common/buffers.py` type hints
- Fixed `stable_baselines3/her/her_replay_buffer.py` type hints
- Buffers do no call an additional `.copy()` when storing new transitions
- Fixed `ActorCriticPolicy.extract_features()` signature by adding an optional `features_extractor` argument
- Update dependencies (accept newer Shimmy/Sphinx version and remove `sphinx_autodoc_typehints`)
- Fixed `stable_baselines3/common/off_policy_algorithm.py` type hints
- Fixed `stable_baselines3/common/distributions.py` type hints
- Fixed `stable_baselines3/common/vec_env/vec_normalize.py` type hints
- Fixed `stable_baselines3/common/vec_env/__init__.py` type hints
- Switched to PyTorch 2.1.0 in the CI (fixes type annotations)
- Fixed `stable_baselines3/common/policies.py` type hints
- Switched to `mypy` only for checking types
- Added tests to check consistency when saving/loading files

### Documentation:

- Updated RL Tips and Tricks (include recommendation for evaluation, added links to DroQ, ARS and SBX).
- Fixed various typos and grammar mistakes
- Added PokemonRedExperiments to the project page
- Fixed an out-of-date command for installing Atari in examples

## Release 2.1.0 (2023-08-17)

**Float64 actions , Gymnasium 0.29 support and bug fixes**

### Breaking Changes:

- Removed Python 3.7 support
- SB3 now requires PyTorch >= 1.13

### New Features:

- Added Python 3.11 support
- Added Gymnasium 0.29 support (@pseudo-rnd-thoughts)

### [SB3-Contrib]

- Fixed MaskablePPO ignoring `stats_window_size` argument
- Added Python 3.11 support

### [RL Zoo]

- Upgraded to Huggingface-SB3 >= 2.3
- Added Python 3.11 support

### Bug Fixes:

- Relaxed check in logger, that was causing issue on Windows with colorama
- Fixed off-policy algorithms with continuous float64 actions (see #1145) (@tobirohrer)
- Fixed `env_checker.py` warning messages for out of bounds in complex observation spaces (@Gabo-Tor)

### Deprecations:

### Others:

- Updated GitHub issue templates
- Fix typo in gym patch error message (@lukashass)
- Refactor `test_spaces.py` tests

### Documentation:

- Fixed callback example (@BertrandDecoster)
- Fixed policy network example (@kyle-he)
- Added mobile-env as new community project (@stefanbschneider)
- Added \[DeepNetSlice\](<https://github.com/AlexPasqua/DeepNetSlice>) to community projects (@AlexPasqua)

## Release 2.0.0 (2023-06-22)

**Gymnasium support**

:::{warning}
Stable-Baselines3 (SB3) v2.0 will be the last one supporting python 3.7 (end of life in June 2023).
We highly recommended you to upgrade to Python >= 3.8.
:::

### Breaking Changes:

- Switched to Gymnasium as primary backend, Gym 0.21 and 0.26 are still supported via the `shimmy` package (@carlosluis, @arjun-kg, @tlpss)
- The deprecated `online_sampling` argument of `HerReplayBuffer` was removed
- Removed deprecated `stack_observation_space` method of `StackedObservations`
- Renamed environment output observations in `evaluate_policy` to prevent shadowing the input observations during callbacks (@npit)
- Upgraded wrappers and custom environment to Gymnasium
- Refined the `HumanOutputFormat` file check: now it verifies if the object is an instance of `io.TextIOBase` instead of only checking for the presence of a `write` method.
- Because of new Gym API (0.26+), the random seed passed to `vec_env.seed(seed=seed)` will only be effective after then `env.reset()` call.

### New Features:

- Added Gymnasium support (Gym 0.21 and 0.26 are supported via the `shimmy` package)

### [SB3-Contrib]

- Fixed QRDQN update interval for multi envs

### [RL Zoo]

- Gym 0.26+ patches to continue working with pybullet and TimeLimit wrapper
- Renamed `CarRacing-v1` to `CarRacing-v2` in hyperparameters
- Huggingface push to hub now accepts a `--n-timesteps` argument to adjust the length of the video
- Fixed `record_video` steps (before it was stepping in a closed env)
- Dropped Gym 0.21 support

### Bug Fixes:

- Fixed `VecExtractDictObs` does not handle terminal observation (@WeberSamuel)
- Set NumPy version to `>=1.20` due to use of `numpy.typing` (@troiganto)
- Fixed loading DQN changes `target_update_interval` (@tobirohrer)
- Fixed env checker to properly reset the env before calling `step()` when checking
  for `Inf` and `NaN` (@lutogniew)
- Fixed HER `truncate_last_trajectory()` (@lbergmann1)
- Fixed HER desired and achieved goal order in reward computation (@JonathanKuelz)

### Deprecations:

### Others:

- Fixed `stable_baselines3/a2c/*.py` type hints
- Fixed `stable_baselines3/ppo/*.py` type hints
- Fixed `stable_baselines3/sac/*.py` type hints
- Fixed `stable_baselines3/td3/*.py` type hints
- Fixed `stable_baselines3/common/base_class.py` type hints
- Fixed `stable_baselines3/common/logger.py` type hints
- Fixed `stable_baselines3/common/envs/*.py` type hints
- Fixed `stable_baselines3/common/vec_env/vec_monitor|vec_extract_dict_obs|util.py` type hints
- Fixed `stable_baselines3/common/vec_env/base_vec_env.py` type hints
- Fixed `stable_baselines3/common/vec_env/vec_frame_stack.py` type hints
- Fixed `stable_baselines3/common/vec_env/dummy_vec_env.py` type hints
- Fixed `stable_baselines3/common/vec_env/subproc_vec_env.py` type hints
- Upgraded docker images to use mamba/micromamba and CUDA 11.7
- Updated env checker to reflect what subset of Gymnasium is supported and improve GoalEnv checks
- Improve type annotation of wrappers
- Tests envs are now checked too
- Added render test for `VecEnv` and `VecEnvWrapper`
- Update issue templates and env info saved with the model
- Changed `seed()` method return type from `List` to `Sequence`
- Updated env checker doc and requirements for tuple spaces/goal envs

### Documentation:

- Added Deep RL Course link to the Deep RL Resources page
- Added documentation about `VecEnv` API vs Gym API
- Upgraded tutorials to Gymnasium API
- Make it more explicit when using `VecEnv` vs Gym env
- Added UAV_Navigation_DRL_AirSim to the project page (@heleidsn)
- Added `EvalCallback` example (@sidney-tio)
- Update custom env documentation
- Added `pink-noise-rl` to projects page
- Fix custom policy example, `ortho_init` was ignored
- Added SBX page

## Release 1.8.0 (2023-04-07)

**Multi-env HerReplayBuffer, Open RL Benchmark, Improved env checker**

:::{warning}
Stable-Baselines3 (SB3) v1.8.0 will be the last one to use Gym as a backend.
Starting with v2.0.0, Gymnasium will be the default backend (though SB3 will have compatibility layers for Gym envs).
You can find a migration guide here: <https://gymnasium.farama.org/content/migration-guide/>.
If you want to try the SB3 v2.0 alpha version, you can take a look at [PR #1327](https://github.com/DLR-RM/stable-baselines3/pull/1327).
:::

### Breaking Changes:

- Removed shared layers in `mlp_extractor` (@AlexPasqua)
- Refactored `StackedObservations` (it now handles dict obs, `StackedDictObservations` was removed)
- You must now explicitly pass a `features_extractor` parameter when calling `extract_features()`
- Dropped offline sampling for `HerReplayBuffer`
- As `HerReplayBuffer` was refactored to support multiprocessing, previous replay buffer are incompatible with this new version
- `HerReplayBuffer` doesn't require a `max_episode_length` anymore

### New Features:

- Added `repeat_action_probability` argument in `AtariWrapper`.
- Only use `NoopResetEnv` and `MaxAndSkipEnv` when needed in `AtariWrapper`
- Added support for dict/tuple observations spaces for `VecCheckNan`, the check is now active in the `env_checker()` (@DavyMorgan)
- Added multiprocessing support for `HerReplayBuffer`
- `HerReplayBuffer` now supports all datatypes supported by `ReplayBuffer`
- Provide more helpful failure messages when validating the `observation_space` of custom gym environments using `check_env` (@FieteO)
- Added `stats_window_size` argument to control smoothing in rollout logging (@jonasreiher)

### [SB3-Contrib]

- Added warning about potential crashes caused by `check_env` in the `MaskablePPO` docs (@AlexPasqua)
- Fixed `sb3_contrib/qrdqn/*.py` type hints
- Removed shared layers in `mlp_extractor` (@AlexPasqua)

### [RL Zoo]

- [Open RL Benchmark](https://github.com/openrlbenchmark/openrlbenchmark/issues/7)
- Upgraded to new `HerReplayBuffer` implementation that supports multiple envs
- Removed `TimeFeatureWrapper` for Panda and Fetch envs, as the new replay buffer should handle timeout.
- Tuned hyperparameters for RecurrentPPO on Swimmer
- Documentation is now built using Sphinx and hosted on read the doc
- Removed `use_auth_token` for push to hub util
- Reverted from v3 to v2 for HumanoidStandup, Reacher, InvertedPendulum and InvertedDoublePendulum since they were not part of the mujoco refactoring (see <https://github.com/openai/gym/pull/1304>)
- Fixed `gym-minigrid` policy (from `MlpPolicy` to `MultiInputPolicy`)
- Replaced deprecated `optuna.suggest_loguniform(...)` by `optuna.suggest_float(..., log=True)`
- Switched to `ruff` and `pyproject.toml`
- Removed `online_sampling` and `max_episode_length` argument when using `HerReplayBuffer`

### Bug Fixes:

- Fixed Atari wrapper that missed the reset condition (@luizapozzobon)
- Added the argument `dtype` (default to `float32`) to the noise for consistency with gym action (@sidney-tio)
- Fixed PPO train/n_updates metric not accounting for early stopping (@adamfrly)
- Fixed loading of normalized image-based environments
- Fixed `DictRolloutBuffer.add` with multidimensional action space (@younik)

### Deprecations:

### Others:

- Fixed `tests/test_tensorboard.py` type hint
- Fixed `tests/test_vec_normalize.py` type hint
- Fixed `stable_baselines3/common/monitor.py` type hint
- Added tests for StackedObservations
- Removed Gitlab CI file
- Moved from `setup.cg` to `pyproject.toml` configuration file
- Switched from `flake8` to `ruff`
- Upgraded AutoROM to latest version
- Fixed `stable_baselines3/dqn/*.py` type hints
- Added `extra_no_roms` option for package installation without Atari Roms

### Documentation:

- Renamed `load_parameters` to `set_parameters` (@DavyMorgan)
- Clarified documentation about subproc multiprocessing for A2C (@Bonifatius94)
- Fixed typo in `A2C` docstring (@AlexPasqua)
- Renamed timesteps to episodes for `log_interval` description (@theSquaredError)
- Removed note about gif creation for Atari games (@harveybellini)
- Added information about default network architecture
- Update information about Gymnasium support

## Release 1.7.0 (2023-01-10)

:::{warning}
Shared layers in MLP policy (`mlp_extractor`) are now deprecated for PPO, A2C and TRPO.
This feature will be removed in SB3 v1.8.0 and the behavior of `net_arch=[64, 64]`
will create **separate** networks with the same architecture, to be consistent with the off-policy algorithms.
:::

:::{note}
A2C and PPO saved with SB3 < 1.7.0 will show a warning about
missing keys in the state dict when loaded with SB3 >= 1.7.0.
To suppress the warning, simply save the model again.
You can find more info in [issue #1233](https://github.com/DLR-RM/stable-baselines3/issues/1233)
:::

### Breaking Changes:

- Removed deprecated `create_eval_env`, `eval_env`, `eval_log_path`, `n_eval_episodes` and `eval_freq` parameters,
  please use an `EvalCallback` instead
- Removed deprecated `sde_net_arch` parameter
- Removed `ret` attributes in `VecNormalize`, please use `returns` instead
- `VecNormalize` now updates the observation space when normalizing images

### New Features:

- Introduced mypy type checking
- Added option to have non-shared features extractor between actor and critic in on-policy algorithms (@AlexPasqua)
- Added `with_bias` argument to `create_mlp`
- Added support for multidimensional `spaces.MultiBinary` observations
- Features extractors now properly support unnormalized image-like observations (3D tensor)
  when passing `normalize_images=False`
- Added `normalized_image` parameter to `NatureCNN` and `CombinedExtractor`
- Added support for Python 3.10

### [SB3-Contrib]

- Fixed a bug in `RecurrentPPO` where the lstm states where incorrectly reshaped for `n_lstm_layers > 1` (thanks @kolbytn)
- Fixed `RuntimeError: rnn: hx is not contiguous` while predicting terminal values for `RecurrentPPO` when `n_lstm_layers > 1`

### [RL Zoo]

- Added support for python file for configuration
- Added `monitor_kwargs` parameter

### Bug Fixes:

- Fixed `ProgressBarCallback` under-reporting (@dominicgkerr)
- Fixed return type of `evaluate_actions` in `ActorCritcPolicy` to reflect that entropy is an optional tensor (@Rocamonde)
- Fixed type annotation of `policy` in `BaseAlgorithm` and `OffPolicyAlgorithm`
- Allowed model trained with Python 3.7 to be loaded with Python 3.8+ without the `custom_objects` workaround
- Raise an error when the same gym environment instance is passed as separate environments when creating a vectorized environment with more than one environment. (@Rocamonde)
- Fix type annotation of `model` in `evaluate_policy`
- Fixed `Self` return type using `TypeVar`
- Fixed the env checker, the key was not passed when checking images from Dict observation space
- Fixed `normalize_images` which was not passed to parent class in some cases
- Fixed `load_from_vector` that was broken with newer PyTorch version when passing PyTorch tensor

### Deprecations:

- You should now explicitly pass a `features_extractor` parameter when calling `extract_features()`
- Deprecated shared layers in `MlpExtractor` (@AlexPasqua)

### Others:

- Used issue forms instead of issue templates
- Updated the PR template to associate each PR with its peer in RL-Zoo3 and SB3-Contrib
- Fixed flake8 config to be compatible with flake8 6+
- Goal-conditioned environments are now characterized by the availability of the `compute_reward` method, rather than by their inheritance to `gym.GoalEnv`
- Replaced `CartPole-v0` by `CartPole-v1` is tests
- Fixed `tests/test_distributions.py` type hints
- Fixed `stable_baselines3/common/type_aliases.py` type hints
- Fixed `stable_baselines3/common/torch_layers.py` type hints
- Fixed `stable_baselines3/common/env_util.py` type hints
- Fixed `stable_baselines3/common/preprocessing.py` type hints
- Fixed `stable_baselines3/common/atari_wrappers.py` type hints
- Fixed `stable_baselines3/common/vec_env/vec_check_nan.py` type hints
- Exposed modules in `__init__.py` with the `__all__` attribute (@ZikangXiong)
- Upgraded GitHub CI/setup-python to v4 and checkout to v3
- Set tensors construction directly on the device (~8% speed boost on GPU)
- Monkey-patched `np.bool = bool` so gym 0.21 is compatible with NumPy 1.24+
- Standardized the use of `from gym import spaces`
- Modified `get_system_info` to avoid issue linked to copy-pasting on GitHub issue

### Documentation:

- Updated Hugging Face Integration page (@simoninithomas)
- Changed `env` to `vec_env` when environment is vectorized
- Updated custom policy docs to better explain the `mlp_extractor`'s dimensions (@AlexPasqua)
- Updated custom policy documentation (@athatheo)
- Improved tensorboard callback doc
- Clarify doc when using image-like input
- Added RLeXplore to the project page (@yuanmingqi)

## Release 1.6.2 (2022-10-10)

**Progress bar in the learn() method, RL Zoo3 is now a package**

### Breaking Changes:

### New Features:

- Added `progress_bar` argument in the `learn()` method, displayed using TQDM and rich packages
- Added progress bar callback
- The [RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) can now be installed as a package (`pip install rl_zoo3`)

### [SB3-Contrib]

### [RL Zoo]

- RL Zoo is now a python package and can be installed using `pip install rl_zoo3`

### Bug Fixes:

- `self.num_timesteps` was initialized properly only after the first call to `on_step()` for callbacks
- Set importlib-metadata version to `~=4.13` to be compatible with `gym=0.21`

### Deprecations:

- Added deprecation warning if parameters `eval_env`, `eval_freq` or `create_eval_env` are used (see #925) (@tobirohrer)

### Others:

- Fixed type hint of the `env_id` parameter in `make_vec_env` and `make_atari_env` (@AlexPasqua)

### Documentation:

- Extended docstring of the `wrapper_class` parameter in `make_vec_env` (@AlexPasqua)

## Release 1.6.1 (2022-09-29)

**Bug fix release**

### Breaking Changes:

- Switched minimum tensorboard version to 2.9.1

### New Features:

- Support logging hyperparameters to tensorboard (@timothe-chaumont)
- Added checkpoints for replay buffer and `VecNormalize` statistics (@anand-bala)
- Added option for `Monitor` to append to existing file instead of overriding (@sidney-tio)
- The env checker now raises an error when using dict observation spaces and observation keys don't match observation space keys

### [SB3-Contrib]

- Fixed the issue of wrongly passing policy arguments when using `CnnLstmPolicy` or `MultiInputLstmPolicy` with `RecurrentPPO` (@mlodel)

### Bug Fixes:

- Fixed issue where `PPO` gives NaN if rollout buffer provides a batch of size 1 (@hughperkins)
- Fixed the issue that `predict` does not always return action as `np.ndarray` (@qgallouedec)
- Fixed division by zero error when computing FPS when a small number of time has elapsed in operating systems with low-precision timers.
- Added multidimensional action space support (@qgallouedec)
- Fixed missing verbose parameter passing in the `EvalCallback` constructor (@burakdmb)
- Fixed the issue that when updating the target network in DQN, SAC, TD3, the `running_mean` and `running_var` properties of batch norm layers are not updated (@honglu2875)
- Fixed incorrect type annotation of the replay_buffer_class argument in `common.OffPolicyAlgorithm` initializer, where an instance instead of a class was required (@Rocamonde)
- Fixed loading saved model with different number of environments
- Removed `forward()` abstract method declaration from `common.policies.BaseModel` (already defined in `torch.nn.Module`) to fix type errors in subclasses (@Rocamonde)
- Fixed the return type of `.load()` and `.learn()` methods in `BaseAlgorithm` so that they now use `TypeVar` (@Rocamonde)
- Fixed an issue where keys with different tags but the same key raised an error in `common.logger.HumanOutputFormat` (@Rocamonde and @AdamGleave)
- Set importlib-metadata version to `~=4.13`

### Deprecations:

### Others:

- Fixed `DictReplayBuffer.next_observations` typing (@qgallouedec)
- Added support for `device="auto"` in buffers and made it default (@qgallouedec)
- Updated `ResultsWriter` (used internally by `Monitor` wrapper) to automatically create missing directories when `filename` is a path (@dominicgkerr)

### Documentation:

- Added an example of callback that logs hyperparameters to tensorboard. (@timothe-chaumont)
- Fixed typo in docstring "nature" -> "Nature" (@Melanol)
- Added info on split tensorboard logs into (@Melanol)
- Fixed typo in ppo doc (@francescoluciano)
- Fixed typo in install doc(@jlp-ue)
- Clarified and standardized verbosity documentation
- Added link to a GitHub issue in the custom policy documentation (@AlexPasqua)
- Update doc on exporting models (fixes and added torch jit)
- Fixed typos (@Akhilez)
- Standardized the use of `"` for string representation in documentation

## Release 1.6.0 (2022-07-11)

**Recurrent PPO (PPO LSTM), better defaults for learning from pixels with SAC/TD3**

### Breaking Changes:

- Changed the way policy "aliases" are handled ("MlpPolicy", "CnnPolicy", ...), removing the former
  `register_policy` helper, `policy_base` parameter and using `policy_aliases` static attributes instead (@Gregwar)
- SB3 now requires PyTorch >= 1.11
- Changed the default network architecture when using `CnnPolicy` or `MultiInputPolicy` with SAC or DDPG/TD3,
  `share_features_extractor` is now set to False by default and the `net_arch=[256, 256]` (instead of `net_arch=[]` that was before)

### New Features:

### [SB3-Contrib]

- Added Recurrent PPO (PPO LSTM). See <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/53>

### Bug Fixes:

- Fixed saving and loading large policies greater than 2GB (@jkterry1, @ycheng517)
- Fixed final goal selection strategy that did not sample the final achieved goal (@qgallouedec)
- Fixed a bug with special characters in the tensorboard log name (@quantitative-technologies)
- Fixed a bug in `DummyVecEnv`'s and `SubprocVecEnv`'s seeding function. None value was unchecked (@ScheiklP)
- Fixed a bug where `EvalCallback` would crash when trying to synchronize `VecNormalize` stats when observation normalization was disabled
- Added a check for unbounded actions
- Fixed issues due to newer version of protobuf (tensorboard) and sphinx
- Fix exception causes all over the codebase (@cool-RR)
- Prohibit simultaneous use of optimize_memory_usage and handle_timeout_termination due to a bug (@MWeltevrede)
- Fixed a bug in `kl_divergence` check that would fail when using numpy arrays with MultiCategorical distribution

### Deprecations:

### Others:

- Upgraded to Python 3.7+ syntax using `pyupgrade`
- Removed redundant double-check for nested observations from `BaseAlgorithm._wrap_env` (@TibiGG)

### Documentation:

- Added link to gym doc and gym env checker
- Fix typo in PPO doc (@bcollazo)
- Added link to PPO ICLR blog post
- Added remark about breaking Markov assumption and timeout handling
- Added doc about MLFlow integration via custom logger (@git-thor)
- Updated Huggingface integration doc
- Added copy button for code snippets
- Added doc about EnvPool and Isaac Gym support

## Release 1.5.0 (2022-03-25)

**Bug fixes, early stopping callback**

### Breaking Changes:

- Switched minimum Gym version to 0.21.0

### New Features:

- Added `StopTrainingOnNoModelImprovement` to callback collection (@caburu)
- Makes the length of keys and values in `HumanOutputFormat` configurable,
  depending on desired maximum width of output.
- Allow PPO to turn of advantage normalization (see [PR #763](https://github.com/DLR-RM/stable-baselines3/pull/763)) @vwxyzjn

### [SB3-Contrib]

- coming soon: Cross Entropy Method, see <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/62>

### Bug Fixes:

- Fixed a bug in `VecMonitor`. The monitor did not consider the `info_keywords` during stepping (@ScheiklP)
- Fixed a bug in `HumanOutputFormat`. Distinct keys truncated to the same prefix would overwrite each others value,
  resulting in only one being output. This now raises an error (this should only affect a small fraction of use cases
  with very long keys.)
- Routing all the `nn.Module` calls through implicit rather than explicit forward as per pytorch guidelines (@manuel-delverme)
- Fixed a bug in `VecNormalize` where error occurs when `norm_obs` is set to False for environment with dictionary observation (@buoyancy99)
- Set default `env` argument to `None` in `HerReplayBuffer.sample` (@qgallouedec)
- Fix `batch_size` typing in `DQN` (@qgallouedec)
- Fixed sample normalization in `DictReplayBuffer` (@qgallouedec)

### Deprecations:

### Others:

- Fixed pytest warnings
- Removed parameter `remove_time_limit_termination` in off policy algorithms since it was dead code (@Gregwar)

### Documentation:

- Added doc on Hugging Face integration (@simoninithomas)
- Added furuta pendulum project to project list (@armandpl)
- Fix indentation 2 spaces to 4 spaces in custom env documentation example (@Gautam-J)
- Update MlpExtractor docstring (@gianlucadecola)
- Added explanation of the logger output
- Update `Directly Accessing The Summary Writer` in tensorboard integration (@xy9485)

## Release 1.4.0 (2022-01-18)

*TRPO, ARS and multi env training for off-policy algorithms*

### Breaking Changes:

- Dropped python 3.6 support (as announced in previous release)
- Renamed `mask` argument of the `predict()` method to `episode_start` (used with RNN policies only)
- local variables `action`, `done` and `reward` were renamed to their plural form for offpolicy algorithms (`actions`, `dones`, `rewards`),
  this may affect custom callbacks.
- Removed `episode_reward` field from `RolloutReturn()` type

:::{warning}
An update to the `HER` algorithm is planned to support multi-env training and remove the max episode length constrain.
(see [PR #704](https://github.com/DLR-RM/stable-baselines3/pull/704))
This will be a backward incompatible change (model trained with previous version of `HER` won't work with the new version).
:::

### New Features:

- Added `norm_obs_keys` param for `VecNormalize` wrapper to configure which observation keys to normalize (@kachayev)
- Added experimental support to train off-policy algorithms with multiple envs (note: `HerReplayBuffer` currently not supported)
- Handle timeout termination properly for on-policy algorithms (when using `TimeLimit`)
- Added `skip` option to `VecTransposeImage` to skip transforming the channel order when the heuristic is wrong
- Added `copy()` and `combine()` methods to `RunningMeanStd`

### [SB3-Contrib]

- Added Trust Region Policy Optimization (TRPO) (@cyprienc)
- Added Augmented Random Search (ARS) (@sgillen)
- Coming soon: PPO LSTM, see <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/53>

### Bug Fixes:

- Fixed a bug where `set_env()` with `VecNormalize` would result in an error with off-policy algorithms (thanks @cleversonahum)
- FPS calculation is now performed based on number of steps performed during last `learn` call, even when `reset_num_timesteps` is set to `False` (@kachayev)
- Fixed evaluation script for recurrent policies (experimental feature in SB3 contrib)
- Fixed a bug where the observation would be incorrectly detected as non-vectorized instead of throwing an error
- The env checker now properly checks and warns about potential issues for continuous action spaces when the boundaries are too small or when the dtype is not float32
- Fixed a bug in `VecFrameStack` with channel first image envs, where the terminal observation would be wrongly created.

### Deprecations:

### Others:

- Added a warning in the env checker when not using `np.float32` for continuous actions
- Improved test coverage and error message when checking shape of observation
- Added `newline="\n"` when opening CSV monitor files so that each line ends with `\r\n` instead of `\r\r\n` on Windows while Linux environments are not affected (@hsuehch)
- Fixed `device` argument inconsistency (@qgallouedec)

### Documentation:

- Add drivergym to projects page (@theDebugger811)
- Add highway-env to projects page (@eleurent)
- Add tactile-gym to projects page (@ac-93)
- Fix indentation in the RL tips page (@cove9988)
- Update GAE computation docstring
- Add documentation on exporting to TFLite/Coral
- Added JMLR paper and updated citation
- Added link to RL Tips and Tricks video
- Updated `BaseAlgorithm.load` docstring (@Demetrio92)
- Added a note on `load` behavior in the examples (@Demetrio92)
- Updated SB3 Contrib doc
- Fixed A2C and migration guide guidance on how to set epsilon with RMSpropTFLike (@thomasgubler)
- Fixed custom policy documentation (@IperGiove)
- Added doc on Weights & Biases integration

## Release 1.3.0 (2021-10-23)

*Bug fixes and improvements for the user*

:::{warning}
This version will be the last one supporting Python 3.6 (end of life in Dec 2021).
We highly recommended you to upgrade to Python >= 3.7.
:::

### Breaking Changes:

- `sde_net_arch` argument in policies is deprecated and will be removed in a future version.

- `_get_latent` (`ActorCriticPolicy`) was removed

- All logging keys now use underscores instead of spaces (@timokau). Concretely this changes:

  > - `time/total timesteps` to `time/total_timesteps` for off-policy algorithms (PPO and A2C) and the eval callback (on-policy algorithms already used the underscored version),
  > - `rollout/exploration rate` to `rollout/exploration_rate` and
  > - `rollout/success rate` to `rollout/success_rate`.

### New Features:

- Added methods `get_distribution` and `predict_values` for `ActorCriticPolicy` for A2C/PPO/TRPO (@cyprienc)
- Added methods `forward_actor` and `forward_critic` for `MlpExtractor`
- Added `sb3.get_system_info()` helper function to gather version information relevant to SB3 (e.g., Python and PyTorch version)
- Saved models now store system information where agent was trained, and load functions have `print_system_info` parameter to help debugging load issues

### Bug Fixes:

- Fixed `dtype` of observations for `SimpleMultiObsEnv`
- Allow `VecNormalize` to wrap discrete-observation environments to normalize reward
  when observation normalization is disabled
- Fixed a bug where `DQN` would throw an error when using `Discrete` observation and stochastic actions
- Fixed a bug where sub-classed observation spaces could not be used
- Added `force_reset` argument to `load()` and `set_env()` in order to be able to call `learn(reset_num_timesteps=False)` with a new environment

### Deprecations:

### Others:

- Cap gym max version to 0.19 to avoid issues with atari-py and other breaking changes
- Improved error message when using dict observation with the wrong policy
- Improved error message when using `EvalCallback` with two envs not wrapped the same way.
- Added additional infos about supported python version for PyPi in `setup.py`

### Documentation:

- Add Rocket League Gym to list of supported projects (@AechPro)
- Added gym-electric-motor to project page (@wkirgsn)
- Added policy-distillation-baselines to project page (@CUN-bjy)
- Added ONNX export instructions (@batu)
- Update read the doc env (fixed `docutils` issue)
- Fix PPO environment name (@IljaAvadiev)
- Fix custom env doc and add env registration example
- Update algorithms from SB3 Contrib
- Use underscores for numeric literals in examples to improve clarity

## Release 1.2.0 (2021-09-03)

**Hotfix for VecNormalize, training/eval mode support**

### Breaking Changes:

- SB3 now requires PyTorch >= 1.8.1
- `VecNormalize` `ret` attribute was renamed to `returns`

### New Features:

### Bug Fixes:

- Hotfix for `VecNormalize` where the observation filter was not updated at reset (thanks @vwxyzjn)
- Fixed model predictions when using batch normalization and dropout layers by calling `train()` and `eval()` (@davidblom603)
- Fixed model training for DQN, TD3 and SAC so that their target nets always remain in evaluation mode (@ayeright)
- Passing `gradient_steps=0` to an off-policy algorithm will result in no gradient steps being taken (vs as many gradient steps as steps done in the environment
  during the rollout in previous versions)

### Deprecations:

### Others:

- Enabled Python 3.9 in GitHub CI
- Fixed type annotations
- Refactored `predict()` by moving the preprocessing to `obs_to_tensor()` method

### Documentation:

- Updated multiprocessing example
- Added example of `VecEnvWrapper`
- Added a note about logging to tensorboard more often
- Added warning about simplicity of examples and link to RL zoo (@MihaiAnca13)

## Release 1.1.0 (2021-07-01)

**Dict observation support, timeout handling and refactored HER buffer**

### Breaking Changes:

- All customs environments (e.g. the `BitFlippingEnv` or `IdentityEnv`) were moved to `stable_baselines3.common.envs` folder
- Refactored `HER` which is now the `HerReplayBuffer` class that can be passed to any off-policy algorithm
- Handle timeout termination properly for off-policy algorithms (when using `TimeLimit`)
- Renamed `_last_dones` and `dones` to `_last_episode_starts` and `episode_starts` in `RolloutBuffer`.
- Removed `ObsDictWrapper` as `Dict` observation spaces are now supported

```python
her_kwargs = dict(n_sampled_goal=2, goal_selection_strategy="future", online_sampling=True)
# SB3 < 1.1.0
# model = HER("MlpPolicy", env, model_class=SAC, **her_kwargs)
# SB3 >= 1.1.0:
model = SAC("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=her_kwargs)
```

- Updated the KL Divergence estimator in the PPO algorithm to be positive definite and have lower variance (@09tangriro)
- Updated the KL Divergence check in the PPO algorithm to be before the gradient update step rather than after end of epoch (@09tangriro)
- Removed parameter `channels_last` from `is_image_space` as it can be inferred.
- The logger object is now an attribute `model.logger` that be set by the user using `model.set_logger()`
- Changed the signature of `logger.configure` and `utils.configure_logger`, they now return a `Logger` object
- Removed `Logger.CURRENT` and `Logger.DEFAULT`
- Moved `warn(), debug(), log(), info(), dump()` methods to the `Logger` class
- `.learn()` now throws an import error when the user tries to log to tensorboard but the package is not installed

### New Features:

- Added support for single-level `Dict` observation space (@JadenTravnik)
- Added `DictRolloutBuffer` `DictReplayBuffer` to support dictionary observations (@JadenTravnik)
- Added `StackedObservations` and `StackedDictObservations` that are used within `VecFrameStack`
- Added simple 4x4 room Dict test environments
- `HerReplayBuffer` now supports `VecNormalize` when `online_sampling=False`
- Added [VecMonitor](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_monitor.py) and
  [VecExtractDictObs](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_extract_dict_obs.py) wrappers
  to handle gym3-style vectorized environments (@vwxyzjn)
- Ignored the terminal observation if the it is not provided by the environment
  such as the gym3-style vectorized environments. (@vwxyzjn)
- Added policy_base as input to the OnPolicyAlgorithm for more flexibility (@09tangriro)
- Added support for image observation when using `HER`
- Added `replay_buffer_class` and `replay_buffer_kwargs` arguments to off-policy algorithms
- Added `kl_divergence` helper for `Distribution` classes (@09tangriro)
- Added support for vector environments with `num_envs > 1` (@benblack769)
- Added `wrapper_kwargs` argument to `make_vec_env` (@amy12xx)

### Bug Fixes:

- Fixed potential issue when calling off-policy algorithms with default arguments multiple times (the size of the replay buffer would be the same)
- Fixed loading of `ent_coef` for `SAC` and `TQC`, it was not optimized anymore (thanks @Atlis)
- Fixed saving of `A2C` and `PPO` policy when using gSDE (thanks @liusida)
- Fixed a bug where no output would be shown even if `verbose>=1` after passing `verbose=0` once
- Fixed observation buffers dtype in DictReplayBuffer (@c-rizz)
- Fixed EvalCallback tensorboard logs being logged with the incorrect timestep. They are now written with the timestep at which they were recorded. (@skandermoalla)

### Deprecations:

### Others:

- Added `flake8-bugbear` to tests dependencies to find likely bugs
- Updated `env_checker` to reflect support of dict observation spaces
- Added Code of Conduct
- Added tests for GAE and lambda return computation
- Updated distribution entropy test (thanks @09tangriro)
- Added sanity check `batch_size > 1` in PPO to avoid NaN in advantage normalization

### Documentation:

- Added gym pybullet drones project (@JacopoPan)
- Added link to SuperSuit in projects (@justinkterry)
- Fixed DQN example (thanks @ltbd78)
- Clarified channel-first/channel-last recommendation
- Update sphinx environment installation instructions (@tom-doerr)
- Clarified pip installation in Zsh (@tom-doerr)
- Clarified return computation for on-policy algorithms (TD(lambda) estimate was used)
- Added example for using `ProcgenEnv`
- Added note about advanced custom policy example for off-policy algorithms
- Fixed DQN unicode checkmarks
- Updated migration guide (@juancroldan)
- Pinned `docutils==0.16` to avoid issue with rtd theme
- Clarified callback `save_freq` definition
- Added doc on how to pass a custom logger
- Remove recurrent policies from `A2C` docs (@bstee615)

## Release 1.0 (2021-03-15)

**First Major Version**

### Breaking Changes:

- Removed `stable_baselines3.common.cmd_util` (already deprecated), please use `env_util` instead

:::{warning}
A refactoring of the `HER` algorithm is planned together with support for dictionary observations
(see [PR #243](https://github.com/DLR-RM/stable-baselines3/pull/243) and [#351](https://github.com/DLR-RM/stable-baselines3/pull/351))
This will be a backward incompatible change (model trained with previous version of `HER` won't work with the new version).
:::

### New Features:

- Added support for `custom_objects` when loading models

### Bug Fixes:

- Fixed a bug with `DQN` predict method when using `deterministic=False` with image space

### Documentation:

- Fixed examples
- Added new project using SB3: rl_reach (@PierreExeter)
- Added note about slow-down when switching to PyTorch
- Add a note on continual learning and resetting environment

### Others:

- Updated RL-Zoo to reflect the fact that is it more than a collection of trained agents
- Added images to illustrate the training loop and custom policies (created with <https://excalidraw.com/>)
- Updated the custom policy section

## Pre-Release 0.11.1 (2021-02-27)

### Bug Fixes:

- Fixed a bug where `train_freq` was not properly converted when loading a saved model

## Pre-Release 0.11.0 (2021-02-27)

### Breaking Changes:

- `evaluate_policy` now returns rewards/episode lengths from a `Monitor` wrapper if one is present,
  this allows to return the unnormalized reward in the case of Atari games for instance.
- Renamed `common.vec_env.is_wrapped` to `common.vec_env.is_vecenv_wrapped` to avoid confusion
  with the new `is_wrapped()` helper
- Renamed `_get_data()` to `_get_constructor_parameters()` for policies (this affects independent saving/loading of policies)
- Removed `n_episodes_rollout` and merged it with `train_freq`, which now accepts a tuple `(frequency, unit)`:
- `replay_buffer` in `collect_rollout` is no more optional

```python
# SB3 < 0.11.0
# model = SAC("MlpPolicy", env, n_episodes_rollout=1, train_freq=-1)
# SB3 >= 0.11.0:
model = SAC("MlpPolicy", env, train_freq=(1, "episode"))
```

### New Features:

- Add support for `VecFrameStack` to stack on first or last observation dimension, along with
  automatic check for image spaces.
- `VecFrameStack` now has a `channels_order` argument to tell if observations should be stacked
  on the first or last observation dimension (originally always stacked on last).
- Added `common.env_util.is_wrapped` and `common.env_util.unwrap_wrapper` functions for checking/unwrapping
  an environment for specific wrapper.
- Added `env_is_wrapped()` method for `VecEnv` to check if its environments are wrapped
  with given Gym wrappers.
- Added `monitor_kwargs` parameter to `make_vec_env` and `make_atari_env`
- Wrap the environments automatically with a `Monitor` wrapper when possible.
- `EvalCallback` now logs the success rate when available (`is_success` must be present in the info dict)
- Added new wrappers to log images and matplotlib figures to tensorboard. (@zampanteymedio)
- Add support for text records to `Logger`. (@lorenz-h)

### Bug Fixes:

- Fixed bug where code added VecTranspose on channel-first image environments (thanks @qxcv)
- Fixed `DQN` predict method when using single `gym.Env` with `deterministic=False`
- Fixed bug that the arguments order of `explained_variance()` in `ppo.py` and `a2c.py` is not correct (@thisray)
- Fixed bug where full `HerReplayBuffer` leads to an index error. (@megan-klaiber)
- Fixed bug where replay buffer could not be saved if it was too big (> 4 Gb) for python\<3.8 (thanks @hn2)
- Added informative `PPO` construction error in edge-case scenario where `n_steps * n_envs = 1` (size of rollout buffer),
  which otherwise causes downstream breaking errors in training (@decodyng)
- Fixed discrete observation space support when using multiple envs with A2C/PPO (thanks @ardabbour)
- Fixed a bug for TD3 delayed update (the update was off-by-one and not delayed when `train_freq=1`)
- Fixed numpy warning (replaced `np.bool` with `bool`)
- Fixed a bug where `VecNormalize` was not normalizing the terminal observation
- Fixed a bug where `VecTranspose` was not transposing the terminal observation
- Fixed a bug where the terminal observation stored in the replay buffer was not the right one for off-policy algorithms
- Fixed a bug where `action_noise` was not used when using `HER` (thanks @ShangqunYu)

### Deprecations:

### Others:

- Add more issue templates
- Add signatures to callable type annotations (@ernestum)
- Improve error message in `NatureCNN`
- Added checks for supported action spaces to improve clarity of error messages for the user
- Renamed variables in the `train()` method of `SAC`, `TD3` and `DQN` to match SB3-Contrib.
- Updated docker base image to Ubuntu 18.04
- Set tensorboard min version to 2.2.0 (earlier version are apparently not working with PyTorch)
- Added warning for `PPO` when `n_steps * n_envs` is not a multiple of `batch_size` (last mini-batch truncated) (@decodyng)
- Removed some warnings in the tests

### Documentation:

- Updated algorithm table
- Minor docstring improvements regarding rollout (@stheid)
- Fix migration doc for `A2C` (epsilon parameter)
- Fix `clip_range` docstring
- Fix duplicated parameter in `EvalCallback` docstring (thanks @tfederico)
- Added example of learning rate schedule
- Added SUMO-RL as example project (@LucasAlegre)
- Fix docstring of classes in atari_wrappers.py which were inside the constructor (@LucasAlegre)
- Added SB3-Contrib page
- Fix bug in the example code of DQN (@AptX395)
- Add example on how to access the tensorboard summary writer directly. (@lorenz-h)
- Updated migration guide
- Updated custom policy doc (separate policy architecture recommended)
- Added a note about OpenCV headless version
- Corrected typo on documentation (@mschweizer)
- Provide the environment when loading the model in the examples (@lorepieri8)

## Pre-Release 0.10.0 (2020-10-28)

**HER with online and offline sampling, bug fixes for features extraction**

### Breaking Changes:

- **Warning:** Renamed `common.cmd_util` to `common.env_util` for clarity (affects `make_vec_env` and `make_atari_env` functions)

### New Features:

- Allow custom actor/critic network architectures using `net_arch=dict(qf=[400, 300], pi=[64, 64])` for off-policy algorithms (SAC, TD3, DDPG)
- Added Hindsight Experience Replay `HER`. (@megan-klaiber)
- `VecNormalize` now supports `gym.spaces.Dict` observation spaces
- Support logging videos to Tensorboard (@SwamyDev)
- Added `share_features_extractor` argument to `SAC` and `TD3` policies

### Bug Fixes:

- Fix GAE computation for on-policy algorithms (off-by one for the last value) (thanks @Wovchena)
- Fixed potential issue when loading a different environment
- Fix ignoring the exclude parameter when recording logs using json, csv or log as logging format (@SwamyDev)
- Make `make_vec_env` support the `env_kwargs` argument when using an env ID str (@ManifoldFR)
- Fix model creation initializing CUDA even when `device="cpu"` is provided
- Fix `check_env` not checking if the env has a Dict actionspace before calling `_check_nan` (@wmmc88)
- Update the check for spaces unsupported by Stable Baselines 3 to include checks on the action space (@wmmc88)
- Fixed features extractor bug for target network where the same net was shared instead
  of being separate. This bug affects `SAC`, `DDPG` and `TD3` when using `CnnPolicy` (or custom features extractor)
- Fixed a bug when passing an environment when loading a saved model with a `CnnPolicy`, the passed env was not wrapped properly
  (the bug was introduced when implementing `HER` so it should not be present in previous versions)

### Deprecations:

### Others:

- Improved typing coverage
- Improved error messages for unsupported spaces
- Added `.vscode` to the gitignore

### Documentation:

- Added first draft of migration guide
- Added intro to [imitation](https://github.com/HumanCompatibleAI/imitation) library (@shwang)
- Enabled doc for `CnnPolicies`
- Added advanced saving and loading example
- Added base doc for exporting models
- Added example for getting and setting model parameters

## Pre-Release 0.9.0 (2020-10-03)

**Bug fixes, get/set parameters and improved docs**

### Breaking Changes:

- Removed `device` keyword argument of policies; use `policy.to(device)` instead. (@qxcv)
- Rename `BaseClass.get_torch_variables` -> `BaseClass._get_torch_save_params` and `BaseClass.excluded_save_params` -> `BaseClass._excluded_save_params`
- Renamed saved items `tensors` to `pytorch_variables` for clarity
- `make_atari_env`, `make_vec_env` and `set_random_seed` must be imported with (and not directly from `stable_baselines3.common`):

```python
from stable_baselines3.common.cmd_util import make_atari_env, make_vec_env
from stable_baselines3.common.utils import set_random_seed
```

### New Features:

- Added `unwrap_vec_wrapper()` to `common.vec_env` to extract `VecEnvWrapper` if needed
- Added `StopTrainingOnMaxEpisodes` to callback collection (@xicocaio)
- Added `device` keyword argument to `BaseAlgorithm.load()` (@liorcohen5)
- Callbacks have access to rollout collection locals as in SB2. (@PartiallyTyped)
- Added `get_parameters` and `set_parameters` for accessing/setting parameters of the agent
- Added actor/critic loss logging for TD3. (@mloo3)

### Bug Fixes:

- Added `unwrap_vec_wrapper()` to `common.vec_env` to extract `VecEnvWrapper` if needed
- Fixed a bug where the environment was reset twice when using `evaluate_policy`
- Fix logging of `clip_fraction` in PPO (@diditforlulz273)
- Fixed a bug where cuda support was wrongly checked when passing the GPU index, e.g., `device="cuda:0"` (@liorcohen5)
- Fixed a bug when the random seed was not properly set on cuda when passing the GPU index

### Deprecations:

### Others:

- Improve typing coverage of the `VecEnv`
- Fix type annotation of `make_vec_env` (@ManifoldFR)
- Removed `AlreadySteppingError` and `NotSteppingError` that were not used
- Fixed typos in SAC and TD3
- Reorganized functions for clarity in `BaseClass` (save/load functions close to each other, private
  functions at top)
- Clarified docstrings on what is saved and loaded to/from files
- Simplified `save_to_zip_file` function by removing duplicate code
- Store library version along with the saved models
- DQN loss is now logged

### Documentation:

- Added `StopTrainingOnMaxEpisodes` details and example (@xicocaio)
- Updated custom policy section (added custom features extractor example)
- Re-enable `sphinx_autodoc_typehints`
- Updated doc style for type hints and remove duplicated type hints

## Pre-Release 0.8.0 (2020-08-03)

**DQN, DDPG, bug fixes and performance matching for Atari games**

### Breaking Changes:

- `AtariWrapper` and other Atari wrappers were updated to match SB2 ones
- `save_replay_buffer` now receives as argument the file path instead of the folder path (@tirafesi)
- Refactored `Critic` class for `TD3` and `SAC`, it is now called `ContinuousCritic`
  and has an additional parameter `n_critics`
- `SAC` and `TD3` now accept an arbitrary number of critics (e.g. `policy_kwargs=dict(n_critics=3)`)
  instead of only 2 previously

### New Features:

- Added `DQN` Algorithm (@Artemis-Skade)
- Buffer dtype is now set according to action and observation spaces for `ReplayBuffer`
- Added warning when allocation of a buffer may exceed the available memory of the system
  when `psutil` is available
- Saving models now automatically creates the necessary folders and raises appropriate warnings (@PartiallyTyped)
- Refactored opening paths for saving and loading to use strings, pathlib or io.BufferedIOBase (@PartiallyTyped)
- Added `DDPG` algorithm as a special case of `TD3`.
- Introduced `BaseModel` abstract parent for `BasePolicy`, which critics inherit from.

### Bug Fixes:

- Fixed a bug in the `close()` method of `SubprocVecEnv`, causing wrappers further down in the wrapper stack to not be closed. (@NeoExtended)
- Fix target for updating q values in SAC: the entropy term was not conditioned by terminals states
- Use `cloudpickle.load` instead of `pickle.load` in `CloudpickleWrapper`. (@shwang)
- Fixed a bug with orthogonal initialization when `bias=False` in custom policy (@rk37)
- Fixed approximate entropy calculation in PPO and A2C. (@andyshih12)
- Fixed DQN target network sharing features extractor with the main network.
- Fixed storing correct `dones` in on-policy algorithm rollout collection. (@andyshih12)
- Fixed number of filters in final convolutional layer in NatureCNN to match original implementation.

### Deprecations:

### Others:

- Refactored off-policy algorithm to share the same `.learn()` method
- Split the `collect_rollout()` method for off-policy algorithms
- Added `_on_step()` for off-policy base class
- Optimized replay buffer size by removing the need of `next_observations` numpy array
- Optimized polyak updates (1.5-1.95 speedup) through inplace operations (@PartiallyTyped)
- Switch to `black` codestyle and added `make format`, `make check-codestyle` and `commit-checks`
- Ignored errors from newer pytype version
- Added a check when using `gSDE`
- Removed codacy dependency from Dockerfile
- Added `common.sb2_compat.RMSpropTFLike` optimizer, which corresponds closer to the implementation of RMSprop from Tensorflow.

### Documentation:

- Updated notebook links
- Fixed a typo in the section of Enjoy a Trained Agent, in RL Baselines3 Zoo README. (@blurLake)
- Added Unity reacher to the projects page (@koulakis)
- Added PyBullet colab notebook
- Fixed typo in PPO example code (@joeljosephjin)
- Fixed typo in custom policy doc (@RaphaelWag)

## Pre-Release 0.7.0 (2020-06-10)

**Hotfix for PPO/A2C + gSDE, internal refactoring and bug fixes**

### Breaking Changes:

- `render()` method of `VecEnvs` now only accept one argument: `mode`

- Created new file common/torch_layers.py, similar to SB refactoring

  - Contains all PyTorch network layer definitions and features extractors: `MlpExtractor`, `create_mlp`, `NatureCNN`

- Renamed `BaseRLModel` to `BaseAlgorithm` (along with offpolicy and onpolicy variants)

- Moved on-policy and off-policy base algorithms to `common/on_policy_algorithm.py` and `common/off_policy_algorithm.py`, respectively.

- Moved `PPOPolicy` to `ActorCriticPolicy` in common/policies.py

- Moved `PPO` (algorithm class) into `OnPolicyAlgorithm` (`common/on_policy_algorithm.py`), to be shared with A2C

- Moved following functions from `BaseAlgorithm`:

  - `_load_from_file` to `load_from_zip_file` (save_util.py)
  - `_save_to_file_zip` to `save_to_zip_file` (save_util.py)
  - `safe_mean` to `safe_mean` (utils.py)
  - `check_env` to `check_for_correct_spaces` (utils.py. Renamed to avoid confusion with environment checker tools)

- Moved static function `_is_vectorized_observation` from common/policies.py to common/utils.py under name `is_vectorized_observation`.

- Removed `{save,load}_running_average` functions of `VecNormalize` in favor of `load/save`.

- Removed `use_gae` parameter from `RolloutBuffer.compute_returns_and_advantage`.

### New Features:

### Bug Fixes:

- Fixed `render()` method for `VecEnvs`
- Fixed `seed()` method for `SubprocVecEnv`
- Fixed loading on GPU for testing when using gSDE and `deterministic=False`
- Fixed `register_policy` to allow re-registering same policy for same sub-class (i.e. assign same value to same key).
- Fixed a bug where the gradient was passed when using `gSDE` with `PPO`/`A2C`, this does not affect `SAC`

### Deprecations:

### Others:

- Re-enable unsafe `fork` start method in the tests (was causing a deadlock with tensorflow)
- Added a test for seeding `SubprocVecEnv` and rendering
- Fixed reference in NatureCNN (pointed to older version with different network architecture)
- Fixed comments saying "CxWxH" instead of "CxHxW" (same style as in torch docs / commonly used)
- Added bit further comments on register/getting policies ("MlpPolicy", "CnnPolicy").
- Renamed `progress` (value from 1 in start of training to 0 in end) to `progress_remaining`.
- Added `policies.py` files for A2C/PPO, which define MlpPolicy/CnnPolicy (renamed ActorCriticPolicies).
- Added some missing tests for `VecNormalize`, `VecCheckNan` and `PPO`.

### Documentation:

- Added a paragraph on "MlpPolicy"/"CnnPolicy" and policy naming scheme under "Developer Guide"
- Fixed second-level listing in changelog

## Pre-Release 0.6.0 (2020-06-01)

**Tensorboard support, refactored logger**

### Breaking Changes:

- Remove State-Dependent Exploration (SDE) support for `TD3`

- Methods were renamed in the logger:

  - `logkv` -> `record`, `writekvs` -> `write`, `writeseq` -> `write_sequence`,
  - `logkvs` -> `record_dict`, `dumpkvs` -> `dump`,
  - `getkvs` -> `get_log_dict`, `logkv_mean` -> `record_mean`,

### New Features:

- Added env checker (Sync with Stable Baselines)
- Added `VecCheckNan` and `VecVideoRecorder` (Sync with Stable Baselines)
- Added determinism tests
- Added `cmd_util` and `atari_wrappers`
- Added support for `MultiDiscrete` and `MultiBinary` observation spaces (@rolandgvc)
- Added `MultiCategorical` and `Bernoulli` distributions for PPO/A2C (@rolandgvc)
- Added support for logging to tensorboard (@rolandgvc)
- Added `VectorizedActionNoise` for continuous vectorized environments (@PartiallyTyped)
- Log evaluation in the `EvalCallback` using the logger

### Bug Fixes:

- Fixed a bug that prevented model trained on cpu to be loaded on gpu
- Fixed version number that had a new line included
- Fixed weird seg fault in docker image due to FakeImageEnv by reducing screen size
- Fixed `sde_sample_freq` that was not taken into account for SAC
- Pass logger module to `BaseCallback` otherwise they cannot write in the one used by the algorithms

### Deprecations:

### Others:

- Renamed to Stable-Baseline3
- Added Dockerfile
- Sync `VecEnvs` with Stable-Baselines
- Update requirement: `gym>=0.17`
- Added `.readthedoc.yml` file
- Added `flake8` and `make lint` command
- Added Github workflow
- Added warning when passing both `train_freq` and `n_episodes_rollout` to Off-Policy Algorithms

### Documentation:

- Added most documentation (adapted from Stable-Baselines)
- Added link to CONTRIBUTING.md in the README (@kinalmehta)
- Added gSDE project and update docstrings accordingly
- Fix `TD3` example code block

## Pre-Release 0.5.0 (2020-05-05)

**CnnPolicy support for image observations, complete saving/loading for policies**

### Breaking Changes:

- Previous loading of policy weights is broken and replace by the new saving/loading for policy

### New Features:

- Added `optimizer_class` and `optimizer_kwargs` to `policy_kwargs` in order to easily
  customizer optimizers
- Complete independent save/load for policies
- Add `CnnPolicy` and `VecTransposeImage` to support images as input

### Bug Fixes:

- Fixed `reset_num_timesteps` behavior, so `env.reset()` is not called if `reset_num_timesteps=True`
- Fixed `squashed_output` that was not pass to policy constructor for `SAC` and `TD3` (would result in scaled actions for unscaled action spaces)

### Deprecations:

### Others:

- Cleanup rollout return
- Added `get_device` util to manage PyTorch devices
- Added type hints to logger + use f-strings

### Documentation:

## Pre-Release 0.4.0 (2020-02-14)

**Proper pre-processing, independent save/load for policies**

### Breaking Changes:

- Removed CEMRL
- Model saved with previous versions cannot be loaded (because of the pre-preprocessing)

### New Features:

- Add support for `Discrete` observation spaces
- Add saving/loading for policy weights, so the policy can be used without the model

### Bug Fixes:

- Fix type hint for activation functions

### Deprecations:

### Others:

- Refactor handling of observation and action spaces
- Refactored features extraction to have proper preprocessing
- Refactored action distributions

## Pre-Release 0.3.0 (2020-02-14)

**Bug fixes, sync with Stable-Baselines, code cleanup**

### Breaking Changes:

- Removed default seed
- Bump dependencies (PyTorch and Gym)
- `predict()` now returns a tuple to match Stable-Baselines behavior

### New Features:

- Better logging for `SAC` and `PPO`

### Bug Fixes:

- Synced callbacks with Stable-Baselines
- Fixed colors in `results_plotter`
- Fix entropy computation (now summed over action dim)

### Others:

- SAC with SDE now sample only one matrix
- Added `clip_mean` parameter to SAC policy
- Buffers now return `NamedTuple`
- More typing
- Add test for `expln`
- Renamed `learning_rate` to `lr_schedule`
- Add `version.txt`
- Add more tests for distribution

### Documentation:

- Deactivated `sphinx_autodoc_typehints` extension

## Pre-Release 0.2.0 (2020-02-14)

**Python 3.6+ required, type checking, callbacks, doc build**

### Breaking Changes:

- Python 2 support was dropped, Stable Baselines3 now requires Python 3.6 or above
- Return type of `evaluation.evaluate_policy()` has been changed
- Refactored the replay buffer to avoid transformation between PyTorch and NumPy
- Created `OffPolicyRLModel` base class
- Remove deprecated JSON format for `Monitor`

### New Features:

- Add `seed()` method to `VecEnv` class
- Add support for Callback (cf <https://github.com/hill-a/stable-baselines/pull/644>)
- Add methods for saving and loading replay buffer
- Add `extend()` method to the buffers
- Add `get_vec_normalize_env()` to `BaseRLModel` to retrieve `VecNormalize` wrapper when it exists
- Add `results_plotter` from Stable Baselines
- Improve `predict()` method to handle different type of observations (single, vectorized, ...)

### Bug Fixes:

- Fix loading model on CPU that were trained on GPU
- Fix `reset_num_timesteps` that was not used
- Fix entropy computation for squashed Gaussian (approximate it now)
- Fix seeding when using multiple environments (different seed per env)

### Others:

- Add type check
- Converted all format string to f-strings
- Add test for `OrnsteinUhlenbeckActionNoise`
- Add type aliases in `common.type_aliases`

### Documentation:

- fix documentation build

## Pre-Release 0.1.0 (2020-01-20)

**First Release: base algorithms and state-dependent exploration**

### New Features:

- Initial release of A2C, CEM-RL, PPO, SAC and TD3, working only with `Box` input space
- State-Dependent Exploration (SDE) for A2C, PPO, SAC and TD3

## Maintainers

Stable-Baselines3 is currently maintained by [Antonin Raffin] (aka [@araffin]), [Ashley Hill] (aka @hill-a),
[Maximilian Ernestus] (aka @ernestum), [Adam Gleave] ([@AdamGleave]), [Anssi Kanervisto] (aka [@Miffyli])
and [Quentin Gallouédec] (aka @qgallouedec).

## Contributors:

In random order...

Thanks to the maintainers of V2: @hill-a @ernestum @AdamGleave @Miffyli

And all the contributors:
@taymuur @bjmuld @iambenzo @iandanforth @r7vme @brendenpetersen @huvar @abhiskk @JohannesAck
@EliasHasle @mrakgr @Bleyddyn @antoine-galataud @junhyeokahn @AdamGleave @keshaviyengar @tperol
@XMaster96 @kantneel @Pastafarianist @GerardMaggiolino @PatrickWalter214 @yutingsz @sc420 @Aaahh @billtubbs
@Miffyli @dwiel @miguelrass @qxcv @jaberkow @eavelardev @ruifeng96150 @pedrohbtp @srivatsankrishnan @evilsocket
@MarvineGothic @jdossgollin @stheid @SyllogismRXS @rusu24edward @jbulow @Antymon @seheevic @justinkterry @edbeeching
@flodorner @KuKuXia @NeoExtended @PartiallyTyped @mmcenta @richardwu @kinalmehta @rolandgvc @tkelestemur @mloo3
@tirafesi @blurLake @koulakis @joeljosephjin @shwang @rk37 @andyshih12 @RaphaelWag @xicocaio
@diditforlulz273 @liorcohen5 @ManifoldFR @mloo3 @SwamyDev @wmmc88 @megan-klaiber @thisray
@tfederico @hn2 @LucasAlegre @AptX395 @zampanteymedio @fracapuano @JadenTravnik @decodyng @ardabbour @lorenz-h @mschweizer @lorepieri8 @vwxyzjn
@ShangqunYu @PierreExeter @JacopoPan @ltbd78 @tom-doerr @Atlis @liusida @09tangriro @amy12xx @juancroldan
@benblack769 @bstee615 @c-rizz @skandermoalla @MihaiAnca13 @davidblom603 @ayeright @cyprienc
@wkirgsn @AechPro @CUN-bjy @batu @IljaAvadiev @timokau @kachayev @cleversonahum
@eleurent @ac-93 @cove9988 @theDebugger811 @hsuehch @Demetrio92 @thomasgubler @IperGiove @ScheiklP
@simoninithomas @armandpl @manuel-delverme @Gautam-J @gianlucadecola @buoyancy99 @caburu @xy9485
@Gregwar @ycheng517 @quantitative-technologies @bcollazo @git-thor @TibiGG @cool-RR @MWeltevrede
@carlosluis @arjun-kg @tlpss @JonathanKuelz @Gabo-Tor @iwishiwasaneagle
@Melanol @qgallouedec @francescoluciano @jlp-ue @burakdmb @timothe-chaumont @honglu2875
@anand-bala @hughperkins @sidney-tio @AlexPasqua @dominicgkerr @Akhilez @Rocamonde @tobirohrer @ZikangXiong @ReHoss
@DavyMorgan @luizapozzobon @Bonifatius94 @theSquaredError @harveybellini @DavyMorgan @FieteO @jonasreiher @npit @WeberSamuel @troiganto
@lutogniew @lbergmann1 @lukashass @BertrandDecoster @pseudo-rnd-thoughts @stefanbschneider @kyle-he @PatrickHelm @corentinlger
@marekm4 @stagoverflow @rushitnshah @markscsmith @NickLucche @cschindlbeck @peteole @jak3122 @will-maclean
@brn-dev @jmacglashan @kplers @MarcDcls @chrisgao99 @pstahlhofen @akanto @Trenza1ore @JonathanColetti @unexploredtest
@m-abr

[@adamgleave]: https://github.com/adamgleave
[@araffin]: https://github.com/araffin
[@miffyli]: https://github.com/Miffyli
[@qgallouedec]: https://github.com/qgallouedec
[adam gleave]: https://gleave.me/
[anssi kanervisto]: https://github.com/Miffyli
[antonin raffin]: https://araffin.github.io/
[ashley hill]: https://github.com/hill-a
[maximilian ernestus]: https://github.com/ernestum
[quentin gallouédec]: https://gallouedec.com/
[rl zoo]: https://github.com/DLR-RM/rl-baselines3-zoo
[sb3-contrib]: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
[sbx]: https://github.com/araffin/sbx
