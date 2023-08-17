.. _changelog:

Changelog
==========

Release 2.1.0 (2023-08-17)
--------------------------

**Float64 actions , Gymnasium 0.29 support and bug fixes**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed Python 3.7 support
- SB3 now requires PyTorch >= 1.13

New Features:
^^^^^^^^^^^^^
- Added Python 3.11 support
- Added Gymnasium 0.29 support (@pseudo-rnd-thoughts)

`SB3-Contrib`_
^^^^^^^^^^^^^^
- Fixed MaskablePPO ignoring ``stats_window_size`` argument
- Added Python 3.11 support

`RL Zoo`_
^^^^^^^^^
- Upgraded to Huggingface-SB3 >= 2.3
- Added Python 3.11 support


Bug Fixes:
^^^^^^^^^^
- Relaxed check in logger, that was causing issue on Windows with colorama
- Fixed off-policy algorithms with continuous float64 actions (see #1145) (@tobirohrer)
- Fixed ``env_checker.py`` warning messages for out of bounds in complex observation spaces (@Gabo-Tor)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Updated GitHub issue templates
- Fix typo in gym patch error message (@lukashass)
- Refactor ``test_spaces.py`` tests

Documentation:
^^^^^^^^^^^^^^
- Fixed callback example (@BertrandDecoster)
- Fixed policy network example (@kyle-he)
- Added mobile-env as new community project (@stefanbschneider)
- Added [DeepNetSlice](https://github.com/AlexPasqua/DeepNetSlice) to community projects (@AlexPasqua)


Release 2.0.0 (2023-06-22)
--------------------------

**Gymnasium support**

.. warning::

  Stable-Baselines3 (SB3) v2.0 will be the last one supporting python 3.7 (end of life in June 2023).
  We highly recommended you to upgrade to Python >= 3.8.


Breaking Changes:
^^^^^^^^^^^^^^^^^
- Switched to Gymnasium as primary backend, Gym 0.21 and 0.26 are still supported via the ``shimmy`` package (@carlosluis, @arjun-kg, @tlpss)
- The deprecated ``online_sampling`` argument of ``HerReplayBuffer`` was removed
- Removed deprecated ``stack_observation_space`` method of ``StackedObservations``
- Renamed environment output observations in ``evaluate_policy`` to prevent shadowing the input observations during callbacks (@npit)
- Upgraded wrappers and custom environment to Gymnasium
- Refined the ``HumanOutputFormat`` file check: now it verifies if the object is an instance of ``io.TextIOBase`` instead of only checking for the presence of a ``write`` method.
- Because of new Gym API (0.26+), the random seed passed to ``vec_env.seed(seed=seed)`` will only be effective after then ``env.reset()`` call.

New Features:
^^^^^^^^^^^^^
- Added Gymnasium support (Gym 0.21 and 0.26 are supported via the ``shimmy`` package)

`SB3-Contrib`_
^^^^^^^^^^^^^^
- Fixed QRDQN update interval for multi envs


`RL Zoo`_
^^^^^^^^^
- Gym 0.26+ patches to continue working with pybullet and TimeLimit wrapper
- Renamed `CarRacing-v1` to `CarRacing-v2` in hyperparameters
- Huggingface push to hub now accepts a `--n-timesteps` argument to adjust the length of the video
- Fixed `record_video` steps (before it was stepping in a closed env)
- Dropped Gym 0.21 support

Bug Fixes:
^^^^^^^^^^
- Fixed ``VecExtractDictObs`` does not handle terminal observation (@WeberSamuel)
- Set NumPy version to ``>=1.20`` due to use of ``numpy.typing`` (@troiganto)
- Fixed loading DQN changes ``target_update_interval`` (@tobirohrer)
- Fixed env checker to properly reset the env before calling ``step()`` when checking
  for ``Inf`` and ``NaN`` (@lutogniew)
- Fixed HER ``truncate_last_trajectory()`` (@lbergmann1)
- Fixed HER desired and achieved goal order in reward computation (@JonathanKuelz)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Fixed ``stable_baselines3/a2c/*.py`` type hints
- Fixed ``stable_baselines3/ppo/*.py`` type hints
- Fixed ``stable_baselines3/sac/*.py`` type hints
- Fixed ``stable_baselines3/td3/*.py`` type hints
- Fixed ``stable_baselines3/common/base_class.py`` type hints
- Fixed ``stable_baselines3/common/logger.py`` type hints
- Fixed ``stable_baselines3/common/envs/*.py`` type hints
- Fixed ``stable_baselines3/common/vec_env/vec_monitor|vec_extract_dict_obs|util.py`` type hints
- Fixed ``stable_baselines3/common/vec_env/base_vec_env.py`` type hints
- Fixed ``stable_baselines3/common/vec_env/vec_frame_stack.py`` type hints
- Fixed ``stable_baselines3/common/vec_env/dummy_vec_env.py`` type hints
- Fixed ``stable_baselines3/common/vec_env/subproc_vec_env.py`` type hints
- Upgraded docker images to use mamba/micromamba and CUDA 11.7
- Updated env checker to reflect what subset of Gymnasium is supported and improve GoalEnv checks
- Improve type annotation of wrappers
- Tests envs are now checked too
- Added render test for ``VecEnv`` and ``VecEnvWrapper``
- Update issue templates and env info saved with the model
- Changed ``seed()`` method return type from ``List`` to ``Sequence``
- Updated env checker doc and requirements for tuple spaces/goal envs

Documentation:
^^^^^^^^^^^^^^
- Added Deep RL Course link to the Deep RL Resources page
- Added documentation about ``VecEnv`` API vs Gym API
- Upgraded tutorials to Gymnasium API
- Make it more explicit when using ``VecEnv`` vs Gym env
- Added UAV_Navigation_DRL_AirSim to the project page (@heleidsn)
- Added ``EvalCallback`` example (@sidney-tio)
- Update custom env documentation
- Added `pink-noise-rl` to projects page
- Fix custom policy example, ``ortho_init`` was ignored
- Added SBX page


Release 1.8.0 (2023-04-07)
--------------------------

**Multi-env HerReplayBuffer, Open RL Benchmark, Improved env checker**

.. warning::

  Stable-Baselines3 (SB3) v1.8.0 will be the last one to use Gym as a backend.
  Starting with v2.0.0, Gymnasium will be the default backend (though SB3 will have compatibility layers for Gym envs).
  You can find a migration guide here: https://gymnasium.farama.org/content/migration-guide/.
  If you want to try the SB3 v2.0 alpha version, you can take a look at `PR #1327 <https://github.com/DLR-RM/stable-baselines3/pull/1327>`_.


Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed shared layers in ``mlp_extractor`` (@AlexPasqua)
- Refactored ``StackedObservations`` (it now handles dict obs, ``StackedDictObservations`` was removed)
- You must now explicitely pass a ``features_extractor`` parameter when calling ``extract_features()``
- Dropped offline sampling for ``HerReplayBuffer``
- As ``HerReplayBuffer`` was refactored to support multiprocessing, previous replay buffer are incompatible with this new version
- ``HerReplayBuffer`` doesn't require a ``max_episode_length`` anymore

New Features:
^^^^^^^^^^^^^
- Added ``repeat_action_probability`` argument in ``AtariWrapper``.
- Only use ``NoopResetEnv`` and ``MaxAndSkipEnv`` when needed in ``AtariWrapper``
- Added support for dict/tuple observations spaces for ``VecCheckNan``, the check is now active in the ``env_checker()`` (@DavyMorgan)
- Added multiprocessing support for ``HerReplayBuffer``
- ``HerReplayBuffer`` now supports all datatypes supported by ``ReplayBuffer``
- Provide more helpful failure messages when validating the ``observation_space`` of custom gym environments using ``check_env`` (@FieteO)
- Added ``stats_window_size`` argument to control smoothing in rollout logging (@jonasreiher)


`SB3-Contrib`_
^^^^^^^^^^^^^^
- Added warning about potential crashes caused by ``check_env`` in the ``MaskablePPO`` docs (@AlexPasqua)
- Fixed ``sb3_contrib/qrdqn/*.py`` type hints
- Removed shared layers in ``mlp_extractor`` (@AlexPasqua)

`RL Zoo`_
^^^^^^^^^
- `Open RL Benchmark <https://github.com/openrlbenchmark/openrlbenchmark/issues/7>`_
- Upgraded to new `HerReplayBuffer` implementation that supports multiple envs
- Removed `TimeFeatureWrapper` for Panda and Fetch envs, as the new replay buffer should handle timeout.
- Tuned hyperparameters for RecurrentPPO on Swimmer
- Documentation is now built using Sphinx and hosted on read the doc
- Removed `use_auth_token` for push to hub util
- Reverted from v3 to v2 for HumanoidStandup, Reacher, InvertedPendulum and InvertedDoublePendulum since they were not part of the mujoco refactoring (see https://github.com/openai/gym/pull/1304)
- Fixed `gym-minigrid` policy (from `MlpPolicy` to `MultiInputPolicy`)
- Replaced deprecated `optuna.suggest_loguniform(...)` by `optuna.suggest_float(..., log=True)`
- Switched to `ruff` and `pyproject.toml`
- Removed `online_sampling` and `max_episode_length` argument when using `HerReplayBuffer`

Bug Fixes:
^^^^^^^^^^
- Fixed Atari wrapper that missed the reset condition (@luizapozzobon)
- Added the argument ``dtype`` (default to ``float32``) to the noise for consistency with gym action (@sidney-tio)
- Fixed PPO train/n_updates metric not accounting for early stopping (@adamfrly)
- Fixed loading of normalized image-based environments
- Fixed ``DictRolloutBuffer.add`` with multidimensional action space (@younik)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Fixed ``tests/test_tensorboard.py`` type hint
- Fixed ``tests/test_vec_normalize.py`` type hint
- Fixed ``stable_baselines3/common/monitor.py`` type hint
- Added tests for StackedObservations
- Removed Gitlab CI file
- Moved from ``setup.cg`` to ``pyproject.toml`` configuration file
- Switched from ``flake8`` to ``ruff``
- Upgraded AutoROM to latest version
- Fixed ``stable_baselines3/dqn/*.py`` type hints
- Added ``extra_no_roms`` option for package installation without Atari Roms

Documentation:
^^^^^^^^^^^^^^
- Renamed ``load_parameters`` to ``set_parameters`` (@DavyMorgan)
- Clarified documentation about subproc multiprocessing for A2C (@Bonifatius94)
- Fixed typo in ``A2C`` docstring (@AlexPasqua)
- Renamed timesteps to episodes for ``log_interval`` description (@theSquaredError)
- Removed note about gif creation for Atari games (@harveybellini)
- Added information about default network architecture
- Update information about Gymnasium support

Release 1.7.0 (2023-01-10)
--------------------------

.. warning::

  Shared layers in MLP policy (``mlp_extractor``) are now deprecated for PPO, A2C and TRPO.
  This feature will be removed in SB3 v1.8.0 and the behavior of ``net_arch=[64, 64]``
  will create **separate** networks with the same architecture, to be consistent with the off-policy algorithms.


.. note::

  A2C and PPO saved with SB3 < 1.7.0 will show a warning about
  missing keys in the state dict when loaded with SB3 >= 1.7.0.
  To suppress the warning, simply save the model again.
  You can find more info in `issue #1233 <https://github.com/DLR-RM/stable-baselines3/issues/1233>`_


Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed deprecated ``create_eval_env``, ``eval_env``, ``eval_log_path``, ``n_eval_episodes`` and ``eval_freq`` parameters,
  please use an ``EvalCallback`` instead
- Removed deprecated ``sde_net_arch`` parameter
- Removed ``ret`` attributes in ``VecNormalize``, please use ``returns`` instead
- ``VecNormalize`` now updates the observation space when normalizing images

New Features:
^^^^^^^^^^^^^
- Introduced mypy type checking
- Added option to have non-shared features extractor between actor and critic in on-policy algorithms (@AlexPasqua)
- Added ``with_bias`` argument to ``create_mlp``
- Added support for multidimensional ``spaces.MultiBinary`` observations
- Features extractors now properly support unnormalized image-like observations (3D tensor)
  when passing ``normalize_images=False``
- Added ``normalized_image`` parameter to ``NatureCNN`` and ``CombinedExtractor``
- Added support for Python 3.10

`SB3-Contrib`_
^^^^^^^^^^^^^^
- Fixed a bug in ``RecurrentPPO`` where the lstm states where incorrectly reshaped for ``n_lstm_layers > 1`` (thanks @kolbytn)
- Fixed ``RuntimeError: rnn: hx is not contiguous`` while predicting terminal values for ``RecurrentPPO`` when ``n_lstm_layers > 1``

`RL Zoo`_
^^^^^^^^^
- Added support for python file for configuration
- Added ``monitor_kwargs`` parameter

Bug Fixes:
^^^^^^^^^^
- Fixed ``ProgressBarCallback`` under-reporting (@dominicgkerr)
- Fixed return type of ``evaluate_actions`` in ``ActorCritcPolicy`` to reflect that entropy is an optional tensor (@Rocamonde)
- Fixed type annotation of ``policy`` in ``BaseAlgorithm`` and ``OffPolicyAlgorithm``
- Allowed model trained with Python 3.7 to be loaded with Python 3.8+ without the ``custom_objects`` workaround
- Raise an error when the same gym environment instance is passed as separate environments when creating a vectorized environment with more than one environment. (@Rocamonde)
- Fix type annotation of ``model`` in ``evaluate_policy``
- Fixed ``Self`` return type using ``TypeVar``
- Fixed the env checker, the key was not passed when checking images from Dict observation space
- Fixed ``normalize_images`` which was not passed to parent class in some cases
- Fixed ``load_from_vector`` that was broken with newer PyTorch version when passing PyTorch tensor

Deprecations:
^^^^^^^^^^^^^
- You should now explicitely pass a ``features_extractor`` parameter when calling ``extract_features()``
- Deprecated shared layers in ``MlpExtractor`` (@AlexPasqua)

Others:
^^^^^^^
- Used issue forms instead of issue templates
- Updated the PR template to associate each PR with its peer in RL-Zoo3 and SB3-Contrib
- Fixed flake8 config to be compatible with flake8 6+
- Goal-conditioned environments are now characterized by the availability of the ``compute_reward`` method, rather than by their inheritance to ``gym.GoalEnv``
- Replaced ``CartPole-v0`` by ``CartPole-v1`` is tests
- Fixed ``tests/test_distributions.py`` type hints
- Fixed ``stable_baselines3/common/type_aliases.py`` type hints
- Fixed ``stable_baselines3/common/torch_layers.py`` type hints
- Fixed ``stable_baselines3/common/env_util.py`` type hints
- Fixed ``stable_baselines3/common/preprocessing.py`` type hints
- Fixed ``stable_baselines3/common/atari_wrappers.py`` type hints
- Fixed ``stable_baselines3/common/vec_env/vec_check_nan.py`` type hints
- Exposed modules in ``__init__.py`` with the ``__all__`` attribute (@ZikangXiong)
- Upgraded GitHub CI/setup-python to v4 and checkout to v3
- Set tensors construction directly on the device (~8% speed boost on GPU)
- Monkey-patched ``np.bool = bool`` so gym 0.21 is compatible with NumPy 1.24+
- Standardized the use of ``from gym import spaces``
- Modified ``get_system_info`` to avoid issue linked to copy-pasting on GitHub issue

Documentation:
^^^^^^^^^^^^^^
- Updated Hugging Face Integration page (@simoninithomas)
- Changed ``env`` to ``vec_env`` when environment is vectorized
- Updated custom policy docs to better explain the ``mlp_extractor``'s dimensions (@AlexPasqua)
- Updated custom policy documentation (@athatheo)
- Improved tensorboard callback doc
- Clarify doc when using image-like input
- Added RLeXplore to the project page (@yuanmingqi)


Release 1.6.2 (2022-10-10)
--------------------------

**Progress bar in the learn() method, RL Zoo3 is now a package**

Breaking Changes:
^^^^^^^^^^^^^^^^^

New Features:
^^^^^^^^^^^^^
- Added ``progress_bar`` argument in the ``learn()`` method, displayed using TQDM and rich packages
- Added progress bar callback
- The `RL Zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ can now be installed as a package (``pip install rl_zoo3``)

`SB3-Contrib`_
^^^^^^^^^^^^^^

`RL Zoo`_
^^^^^^^^^
- RL Zoo is now a python package and can be installed using ``pip install rl_zoo3``

Bug Fixes:
^^^^^^^^^^
- ``self.num_timesteps`` was initialized properly only after the first call to ``on_step()`` for callbacks
- Set importlib-metadata version to ``~=4.13`` to be compatible with ``gym=0.21``

Deprecations:
^^^^^^^^^^^^^
- Added deprecation warning if parameters ``eval_env``, ``eval_freq`` or ``create_eval_env`` are used (see #925) (@tobirohrer)

Others:
^^^^^^^
- Fixed type hint of the ``env_id`` parameter in ``make_vec_env`` and ``make_atari_env`` (@AlexPasqua)

Documentation:
^^^^^^^^^^^^^^
- Extended docstring of the ``wrapper_class`` parameter in ``make_vec_env`` (@AlexPasqua)

Release 1.6.1 (2022-09-29)
---------------------------

**Bug fix release**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Switched minimum tensorboard version to 2.9.1

New Features:
^^^^^^^^^^^^^
- Support logging hyperparameters to tensorboard (@timothe-chaumont)
- Added checkpoints for replay buffer and ``VecNormalize`` statistics (@anand-bala)
- Added option for ``Monitor`` to append to existing file instead of overriding (@sidney-tio)
- The env checker now raises an error when using dict observation spaces and observation keys don't match observation space keys

`SB3-Contrib`_
^^^^^^^^^^^^^^
- Fixed the issue of wrongly passing policy arguments when using ``CnnLstmPolicy`` or ``MultiInputLstmPolicy`` with ``RecurrentPPO`` (@mlodel)

Bug Fixes:
^^^^^^^^^^
- Fixed issue where ``PPO`` gives NaN if rollout buffer provides a batch of size 1 (@hughperkins)
- Fixed the issue that ``predict`` does not always return action as ``np.ndarray`` (@qgallouedec)
- Fixed division by zero error when computing FPS when a small number of time has elapsed in operating systems with low-precision timers.
- Added multidimensional action space support (@qgallouedec)
- Fixed missing verbose parameter passing in the ``EvalCallback`` constructor (@burakdmb)
- Fixed the issue that when updating the target network in DQN, SAC, TD3, the ``running_mean`` and ``running_var`` properties of batch norm layers are not updated (@honglu2875)
- Fixed incorrect type annotation of the replay_buffer_class argument in ``common.OffPolicyAlgorithm`` initializer, where an instance instead of a class was required (@Rocamonde)
- Fixed loading saved model with different number of environments
- Removed ``forward()`` abstract method declaration from ``common.policies.BaseModel`` (already defined in ``torch.nn.Module``) to fix type errors in subclasses (@Rocamonde)
- Fixed the return type of ``.load()`` and ``.learn()`` methods in ``BaseAlgorithm`` so that they now use ``TypeVar`` (@Rocamonde)
- Fixed an issue where keys with different tags but the same key raised an error in ``common.logger.HumanOutputFormat`` (@Rocamonde and @AdamGleave)
- Set importlib-metadata version to `~=4.13`

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Fixed ``DictReplayBuffer.next_observations`` typing (@qgallouedec)
- Added support for ``device="auto"`` in buffers and made it default (@qgallouedec)
- Updated ``ResultsWriter`` (used internally by ``Monitor`` wrapper) to automatically create missing directories when ``filename`` is a path (@dominicgkerr)

Documentation:
^^^^^^^^^^^^^^
- Added an example of callback that logs hyperparameters to tensorboard. (@timothe-chaumont)
- Fixed typo in docstring "nature" -> "Nature" (@Melanol)
- Added info on split tensorboard logs into (@Melanol)
- Fixed typo in ppo doc (@francescoluciano)
- Fixed typo in install doc(@jlp-ue)
- Clarified and standardized verbosity documentation
- Added link to a GitHub issue in the custom policy documentation (@AlexPasqua)
- Update doc on exporting models (fixes and added torch jit)
- Fixed typos (@Akhilez)
- Standardized the use of ``"`` for string representation in documentation

Release 1.6.0 (2022-07-11)
---------------------------

**Recurrent PPO (PPO LSTM), better defaults for learning from pixels with SAC/TD3**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Changed the way policy "aliases" are handled ("MlpPolicy", "CnnPolicy", ...), removing the former
  ``register_policy`` helper, ``policy_base`` parameter and using ``policy_aliases`` static attributes instead (@Gregwar)
- SB3 now requires PyTorch >= 1.11
- Changed the default network architecture when using ``CnnPolicy`` or ``MultiInputPolicy`` with SAC or DDPG/TD3,
  ``share_features_extractor`` is now set to False by default and the ``net_arch=[256, 256]`` (instead of ``net_arch=[]`` that was before)

New Features:
^^^^^^^^^^^^^


`SB3-Contrib`_
^^^^^^^^^^^^^^
- Added Recurrent PPO (PPO LSTM). See https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/53


Bug Fixes:
^^^^^^^^^^
- Fixed saving and loading large policies greater than 2GB (@jkterry1, @ycheng517)
- Fixed final goal selection strategy that did not sample the final achieved goal (@qgallouedec)
- Fixed a bug with special characters in the tensorboard log name (@quantitative-technologies)
- Fixed a bug in ``DummyVecEnv``'s and ``SubprocVecEnv``'s seeding function. None value was unchecked (@ScheiklP)
- Fixed a bug where ``EvalCallback`` would crash when trying to synchronize ``VecNormalize`` stats when observation normalization was disabled
- Added a check for unbounded actions
- Fixed issues due to newer version of protobuf (tensorboard) and sphinx
- Fix exception causes all over the codebase (@cool-RR)
- Prohibit simultaneous use of optimize_memory_usage and handle_timeout_termination due to a bug (@MWeltevrede)
- Fixed a bug in ``kl_divergence`` check that would fail when using numpy arrays with MultiCategorical distribution

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Upgraded to Python 3.7+ syntax using ``pyupgrade``
- Removed redundant double-check for nested observations from ``BaseAlgorithm._wrap_env`` (@TibiGG)

Documentation:
^^^^^^^^^^^^^^
- Added link to gym doc and gym env checker
- Fix typo in PPO doc (@bcollazo)
- Added link to PPO ICLR blog post
- Added remark about breaking Markov assumption and timeout handling
- Added doc about MLFlow integration via custom logger (@git-thor)
- Updated Huggingface integration doc
- Added copy button for code snippets
- Added doc about EnvPool and Isaac Gym support


Release 1.5.0 (2022-03-25)
---------------------------

**Bug fixes, early stopping callback**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Switched minimum Gym version to 0.21.0

New Features:
^^^^^^^^^^^^^
- Added ``StopTrainingOnNoModelImprovement`` to callback collection (@caburu)
- Makes the length of keys and values in ``HumanOutputFormat`` configurable,
  depending on desired maximum width of output.
- Allow PPO to turn of advantage normalization (see `PR #763 <https://github.com/DLR-RM/stable-baselines3/pull/763>`_) @vwxyzjn

`SB3-Contrib`_
^^^^^^^^^^^^^^
- coming soon: Cross Entropy Method, see https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/62

Bug Fixes:
^^^^^^^^^^
- Fixed a bug in ``VecMonitor``. The monitor did not consider the ``info_keywords`` during stepping (@ScheiklP)
- Fixed a bug in ``HumanOutputFormat``. Distinct keys truncated to the same prefix would overwrite each others value,
  resulting in only one being output. This now raises an error (this should only affect a small fraction of use cases
  with very long keys.)
- Routing all the ``nn.Module`` calls through implicit rather than explict forward as per pytorch guidelines (@manuel-delverme)
- Fixed a bug in ``VecNormalize`` where error occurs when ``norm_obs`` is set to False for environment with dictionary observation  (@buoyancy99)
- Set default ``env`` argument to ``None`` in ``HerReplayBuffer.sample`` (@qgallouedec)
- Fix ``batch_size`` typing in ``DQN`` (@qgallouedec)
- Fixed sample normalization in ``DictReplayBuffer`` (@qgallouedec)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Fixed pytest warnings
- Removed parameter ``remove_time_limit_termination`` in off policy algorithms since it was dead code (@Gregwar)

Documentation:
^^^^^^^^^^^^^^
- Added doc on Hugging Face integration (@simoninithomas)
- Added furuta pendulum project to project list (@armandpl)
- Fix indentation 2 spaces to 4 spaces in custom env documentation example (@Gautam-J)
- Update MlpExtractor docstring (@gianlucadecola)
- Added explanation of the logger output
- Update ``Directly Accessing The Summary Writer`` in tensorboard integration (@xy9485)

Release 1.4.0 (2022-01-18)
---------------------------

*TRPO, ARS and multi env training for off-policy algorithms*

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Dropped python 3.6 support (as announced in previous release)
- Renamed ``mask`` argument of the ``predict()`` method to ``episode_start`` (used with RNN policies only)
- local variables ``action``, ``done`` and ``reward`` were renamed to their plural form for offpolicy algorithms (``actions``, ``dones``, ``rewards``),
  this may affect custom callbacks.
- Removed ``episode_reward`` field from ``RolloutReturn()`` type


.. warning::

    An update to the ``HER`` algorithm is planned to support multi-env training and remove the max episode length constrain.
    (see `PR #704 <https://github.com/DLR-RM/stable-baselines3/pull/704>`_)
    This will be a backward incompatible change (model trained with previous version of ``HER`` won't work with the new version).



New Features:
^^^^^^^^^^^^^
- Added ``norm_obs_keys`` param for ``VecNormalize`` wrapper to configure which observation keys to normalize (@kachayev)
- Added experimental support to train off-policy algorithms with multiple envs (note: ``HerReplayBuffer`` currently not supported)
- Handle timeout termination properly for on-policy algorithms (when using ``TimeLimit``)
- Added ``skip`` option to ``VecTransposeImage`` to skip transforming the channel order when the heuristic is wrong
- Added ``copy()`` and ``combine()`` methods to ``RunningMeanStd``

`SB3-Contrib`_
^^^^^^^^^^^^^^
- Added Trust Region Policy Optimization (TRPO) (@cyprienc)
- Added Augmented Random Search (ARS) (@sgillen)
- Coming soon: PPO LSTM, see https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/53

Bug Fixes:
^^^^^^^^^^
- Fixed a bug where ``set_env()`` with ``VecNormalize`` would result in an error with off-policy algorithms (thanks @cleversonahum)
- FPS calculation is now performed based on number of steps performed during last ``learn`` call, even when ``reset_num_timesteps`` is set to ``False`` (@kachayev)
- Fixed evaluation script for recurrent policies (experimental feature in SB3 contrib)
- Fixed a bug where the observation would be incorrectly detected as non-vectorized instead of throwing an error
- The env checker now properly checks and warns about potential issues for continuous action spaces when the boundaries are too small or when the dtype is not float32
- Fixed a bug in ``VecFrameStack`` with channel first image envs, where the terminal observation would be wrongly created.

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Added a warning in the env checker when not using ``np.float32`` for continuous actions
- Improved test coverage and error message when checking shape of observation
- Added ``newline="\n"`` when opening CSV monitor files so that each line ends with ``\r\n`` instead of ``\r\r\n`` on Windows while Linux environments are not affected (@hsuehch)
- Fixed ``device`` argument inconsistency (@qgallouedec)

Documentation:
^^^^^^^^^^^^^^
- Add drivergym to projects page (@theDebugger811)
- Add highway-env to projects page (@eleurent)
- Add tactile-gym to projects page (@ac-93)
- Fix indentation in the RL tips page (@cove9988)
- Update GAE computation docstring
- Add documentation on exporting to TFLite/Coral
- Added JMLR paper and updated citation
- Added link to RL Tips and Tricks video
- Updated ``BaseAlgorithm.load`` docstring (@Demetrio92)
- Added a note on ``load`` behavior in the examples (@Demetrio92)
- Updated SB3 Contrib doc
- Fixed A2C and migration guide guidance on how to set epsilon with RMSpropTFLike (@thomasgubler)
- Fixed custom policy documentation (@IperGiove)
- Added doc on Weights & Biases integration

Release 1.3.0 (2021-10-23)
---------------------------

*Bug fixes and improvements for the user*

.. warning::

  This version will be the last one supporting Python 3.6 (end of life in Dec 2021).
  We highly recommended you to upgrade to Python >= 3.7.


Breaking Changes:
^^^^^^^^^^^^^^^^^
- ``sde_net_arch`` argument in policies is deprecated and will be removed in a future version.
- ``_get_latent`` (``ActorCriticPolicy``) was removed
- All logging keys now use underscores instead of spaces (@timokau). Concretely this changes:

    - ``time/total timesteps`` to ``time/total_timesteps`` for off-policy algorithms (PPO and A2C) and the eval callback (on-policy algorithms already used the underscored version),
    - ``rollout/exploration rate`` to ``rollout/exploration_rate`` and
    - ``rollout/success rate`` to ``rollout/success_rate``.


New Features:
^^^^^^^^^^^^^
- Added methods ``get_distribution`` and ``predict_values`` for ``ActorCriticPolicy`` for A2C/PPO/TRPO (@cyprienc)
- Added methods ``forward_actor`` and ``forward_critic`` for ``MlpExtractor``
- Added ``sb3.get_system_info()`` helper function to gather version information relevant to SB3 (e.g., Python and PyTorch version)
- Saved models now store system information where agent was trained, and load functions have ``print_system_info`` parameter to help debugging load issues

Bug Fixes:
^^^^^^^^^^
- Fixed ``dtype`` of observations for ``SimpleMultiObsEnv``
- Allow `VecNormalize` to wrap discrete-observation environments to normalize reward
  when observation normalization is disabled
- Fixed a bug where ``DQN`` would throw an error when using ``Discrete`` observation and stochastic actions
- Fixed a bug where sub-classed observation spaces could not be used
- Added ``force_reset`` argument to ``load()`` and ``set_env()`` in order to be able to call ``learn(reset_num_timesteps=False)`` with a new environment

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Cap gym max version to 0.19 to avoid issues with atari-py and other breaking changes
- Improved error message when using dict observation with the wrong policy
- Improved error message when using ``EvalCallback`` with two envs not wrapped the same way.
- Added additional infos about supported python version for PyPi in ``setup.py``

Documentation:
^^^^^^^^^^^^^^
- Add Rocket League Gym to list of supported projects (@AechPro)
- Added gym-electric-motor to project page (@wkirgsn)
- Added policy-distillation-baselines to project page (@CUN-bjy)
- Added ONNX export instructions (@batu)
- Update read the doc env (fixed ``docutils`` issue)
- Fix PPO environment name (@IljaAvadiev)
- Fix custom env doc and add env registration example
- Update algorithms from SB3 Contrib
- Use underscores for numeric literals in examples to improve clarity

Release 1.2.0 (2021-09-03)
---------------------------

**Hotfix for VecNormalize, training/eval mode support**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- SB3 now requires PyTorch >= 1.8.1
- ``VecNormalize`` ``ret`` attribute was renamed to ``returns``

New Features:
^^^^^^^^^^^^^

Bug Fixes:
^^^^^^^^^^
- Hotfix for ``VecNormalize`` where the observation filter was not updated at reset (thanks @vwxyzjn)
- Fixed model predictions when using batch normalization and dropout layers by calling ``train()`` and ``eval()`` (@davidblom603)
- Fixed model training for DQN, TD3 and SAC so that their target nets always remain in evaluation mode (@ayeright)
- Passing ``gradient_steps=0`` to an off-policy algorithm will result in no gradient steps being taken (vs as many gradient steps as steps done in the environment
  during the rollout in previous versions)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Enabled Python 3.9 in GitHub CI
- Fixed type annotations
- Refactored ``predict()`` by moving the preprocessing to ``obs_to_tensor()`` method

Documentation:
^^^^^^^^^^^^^^
- Updated multiprocessing example
- Added example of ``VecEnvWrapper``
- Added a note about logging to tensorboard more often
- Added warning about simplicity of examples and link to RL zoo (@MihaiAnca13)


Release 1.1.0 (2021-07-01)
---------------------------

**Dict observation support, timeout handling and refactored HER buffer**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- All customs environments (e.g. the ``BitFlippingEnv`` or ``IdentityEnv``) were moved to ``stable_baselines3.common.envs`` folder
- Refactored ``HER`` which is now the ``HerReplayBuffer`` class that can be passed to any off-policy algorithm
- Handle timeout termination properly for off-policy algorithms (when using ``TimeLimit``)
- Renamed ``_last_dones`` and ``dones`` to ``_last_episode_starts`` and ``episode_starts`` in ``RolloutBuffer``.
- Removed ``ObsDictWrapper`` as ``Dict`` observation spaces are now supported

.. code-block:: python

  her_kwargs = dict(n_sampled_goal=2, goal_selection_strategy="future", online_sampling=True)
  # SB3 < 1.1.0
  # model = HER("MlpPolicy", env, model_class=SAC, **her_kwargs)
  # SB3 >= 1.1.0:
  model = SAC("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=her_kwargs)

- Updated the KL Divergence estimator in the PPO algorithm to be positive definite and have lower variance (@09tangriro)
- Updated the KL Divergence check in the PPO algorithm to be before the gradient update step rather than after end of epoch (@09tangriro)
- Removed parameter ``channels_last`` from ``is_image_space`` as it can be inferred.
- The logger object is now an attribute ``model.logger`` that be set by the user using ``model.set_logger()``
- Changed the signature of ``logger.configure`` and ``utils.configure_logger``, they now return a ``Logger`` object
- Removed ``Logger.CURRENT`` and ``Logger.DEFAULT``
- Moved ``warn(), debug(), log(), info(), dump()`` methods to the ``Logger`` class
- ``.learn()`` now throws an import error when the user tries to log to tensorboard but the package is not installed

New Features:
^^^^^^^^^^^^^
- Added support for single-level ``Dict`` observation space (@JadenTravnik)
- Added ``DictRolloutBuffer`` ``DictReplayBuffer`` to support dictionary observations (@JadenTravnik)
- Added ``StackedObservations`` and ``StackedDictObservations`` that are used within ``VecFrameStack``
- Added simple 4x4 room Dict test environments
- ``HerReplayBuffer`` now supports ``VecNormalize`` when ``online_sampling=False``
- Added `VecMonitor <https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_monitor.py>`_ and
  `VecExtractDictObs <https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_extract_dict_obs.py>`_ wrappers
  to handle gym3-style vectorized environments (@vwxyzjn)
- Ignored the terminal observation if the it is not provided by the environment
  such as the gym3-style vectorized environments. (@vwxyzjn)
- Added policy_base as input to the OnPolicyAlgorithm for more flexibility (@09tangriro)
- Added support for image observation when using ``HER``
- Added ``replay_buffer_class`` and ``replay_buffer_kwargs`` arguments to off-policy algorithms
- Added ``kl_divergence`` helper for ``Distribution`` classes (@09tangriro)
- Added support for vector environments with ``num_envs > 1`` (@benblack769)
- Added ``wrapper_kwargs`` argument to ``make_vec_env`` (@amy12xx)

Bug Fixes:
^^^^^^^^^^
- Fixed potential issue when calling off-policy algorithms with default arguments multiple times (the size of the replay buffer would be the same)
- Fixed loading of ``ent_coef`` for ``SAC`` and ``TQC``, it was not optimized anymore (thanks @Atlis)
- Fixed saving of ``A2C`` and ``PPO`` policy when using gSDE (thanks @liusida)
- Fixed a bug where no output would be shown even if ``verbose>=1`` after passing ``verbose=0`` once
- Fixed observation buffers dtype in DictReplayBuffer (@c-rizz)
- Fixed EvalCallback tensorboard logs being logged with the incorrect timestep. They are now written with the timestep at which they were recorded. (@skandermoalla)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Added ``flake8-bugbear`` to tests dependencies to find likely bugs
- Updated ``env_checker`` to reflect support of dict observation spaces
- Added Code of Conduct
- Added tests for GAE and lambda return computation
- Updated distribution entropy test (thanks @09tangriro)
- Added sanity check ``batch_size > 1`` in PPO to avoid NaN in advantage normalization

Documentation:
^^^^^^^^^^^^^^
- Added gym pybullet drones project (@JacopoPan)
- Added link to SuperSuit in projects (@justinkterry)
- Fixed DQN example (thanks @ltbd78)
- Clarified channel-first/channel-last recommendation
- Update sphinx environment installation instructions (@tom-doerr)
- Clarified pip installation in Zsh (@tom-doerr)
- Clarified return computation for on-policy algorithms (TD(lambda) estimate was used)
- Added example for using ``ProcgenEnv``
- Added note about advanced custom policy example for off-policy algorithms
- Fixed DQN unicode checkmarks
- Updated migration guide (@juancroldan)
- Pinned ``docutils==0.16`` to avoid issue with rtd theme
- Clarified callback ``save_freq`` definition
- Added doc on how to pass a custom logger
- Remove recurrent policies from ``A2C`` docs (@bstee615)


Release 1.0 (2021-03-15)
------------------------

**First Major Version**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed ``stable_baselines3.common.cmd_util`` (already deprecated), please use ``env_util`` instead

.. warning::

    A refactoring of the ``HER`` algorithm is planned together with support for dictionary observations
    (see `PR #243 <https://github.com/DLR-RM/stable-baselines3/pull/243>`_ and `#351 <https://github.com/DLR-RM/stable-baselines3/pull/351>`_)
    This will be a backward incompatible change (model trained with previous version of ``HER`` won't work with the new version).


New Features:
^^^^^^^^^^^^^
- Added support for ``custom_objects`` when loading models



Bug Fixes:
^^^^^^^^^^
- Fixed a bug with ``DQN`` predict method when using ``deterministic=False`` with image space

Documentation:
^^^^^^^^^^^^^^
- Fixed examples
- Added new project using SB3: rl_reach (@PierreExeter)
- Added note about slow-down when switching to PyTorch
- Add a note on continual learning and resetting environment

Others:
^^^^^^^
- Updated RL-Zoo to reflect the fact that is it more than a collection of trained agents
- Added images to illustrate the training loop and custom policies (created with https://excalidraw.com/)
- Updated the custom policy section


Pre-Release 0.11.1 (2021-02-27)
-------------------------------

Bug Fixes:
^^^^^^^^^^
- Fixed a bug where ``train_freq`` was not properly converted when loading a saved model



Pre-Release 0.11.0 (2021-02-27)
-------------------------------

Breaking Changes:
^^^^^^^^^^^^^^^^^
- ``evaluate_policy`` now returns rewards/episode lengths from a ``Monitor`` wrapper if one is present,
  this allows to return the unnormalized reward in the case of Atari games for instance.
- Renamed ``common.vec_env.is_wrapped`` to ``common.vec_env.is_vecenv_wrapped`` to avoid confusion
  with the new ``is_wrapped()`` helper
- Renamed ``_get_data()`` to ``_get_constructor_parameters()`` for policies (this affects independent saving/loading of policies)
- Removed ``n_episodes_rollout`` and merged it with ``train_freq``, which now accepts a tuple ``(frequency, unit)``:
- ``replay_buffer`` in ``collect_rollout`` is no more optional

.. code-block:: python

  # SB3 < 0.11.0
  # model = SAC("MlpPolicy", env, n_episodes_rollout=1, train_freq=-1)
  # SB3 >= 0.11.0:
  model = SAC("MlpPolicy", env, train_freq=(1, "episode"))



New Features:
^^^^^^^^^^^^^
- Add support for ``VecFrameStack`` to stack on first or last observation dimension, along with
  automatic check for image spaces.
- ``VecFrameStack`` now has a ``channels_order`` argument to tell if observations should be stacked
  on the first or last observation dimension (originally always stacked on last).
- Added ``common.env_util.is_wrapped`` and ``common.env_util.unwrap_wrapper`` functions for checking/unwrapping
  an environment for specific wrapper.
- Added ``env_is_wrapped()`` method for ``VecEnv`` to check if its environments are wrapped
  with given Gym wrappers.
- Added ``monitor_kwargs`` parameter to ``make_vec_env`` and ``make_atari_env``
- Wrap the environments automatically with a ``Monitor`` wrapper when possible.
- ``EvalCallback`` now logs the success rate when available (``is_success`` must be present in the info dict)
- Added new wrappers to log images and matplotlib figures to tensorboard. (@zampanteymedio)
- Add support for text records to ``Logger``. (@lorenz-h)


Bug Fixes:
^^^^^^^^^^
- Fixed bug where code added VecTranspose on channel-first image environments (thanks @qxcv)
- Fixed ``DQN`` predict method when using single ``gym.Env`` with ``deterministic=False``
- Fixed bug that the arguments order of ``explained_variance()`` in ``ppo.py`` and ``a2c.py`` is not correct (@thisray)
- Fixed bug where full ``HerReplayBuffer`` leads to an index error. (@megan-klaiber)
- Fixed bug where replay buffer could not be saved if it was too big (> 4 Gb) for python<3.8 (thanks @hn2)
- Added informative ``PPO`` construction error in edge-case scenario where ``n_steps * n_envs = 1`` (size of rollout buffer),
  which otherwise causes downstream breaking errors in training (@decodyng)
- Fixed discrete observation space support when using multiple envs with A2C/PPO (thanks @ardabbour)
- Fixed a bug for TD3 delayed update (the update was off-by-one and not delayed when ``train_freq=1``)
- Fixed numpy warning (replaced ``np.bool`` with ``bool``)
- Fixed a bug where ``VecNormalize`` was not normalizing the terminal observation
- Fixed a bug where ``VecTranspose`` was not transposing the terminal observation
- Fixed a bug where the terminal observation stored in the replay buffer was not the right one for off-policy algorithms
- Fixed a bug where ``action_noise`` was not used when using ``HER`` (thanks @ShangqunYu)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Add more issue templates
- Add signatures to callable type annotations (@ernestum)
- Improve error message in ``NatureCNN``
- Added checks for supported action spaces to improve clarity of error messages for the user
- Renamed variables in the ``train()`` method of ``SAC``, ``TD3`` and ``DQN`` to match SB3-Contrib.
- Updated docker base image to Ubuntu 18.04
- Set tensorboard min version to 2.2.0 (earlier version are apparently not working with PyTorch)
- Added warning for ``PPO`` when ``n_steps * n_envs`` is not a multiple of ``batch_size`` (last mini-batch truncated) (@decodyng)
- Removed some warnings in the tests

Documentation:
^^^^^^^^^^^^^^
- Updated algorithm table
- Minor docstring improvements regarding rollout (@stheid)
- Fix migration doc for ``A2C`` (epsilon parameter)
- Fix ``clip_range`` docstring
- Fix duplicated parameter in ``EvalCallback`` docstring (thanks @tfederico)
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


Pre-Release 0.10.0 (2020-10-28)
-------------------------------

**HER with online and offline sampling, bug fixes for features extraction**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- **Warning:** Renamed ``common.cmd_util`` to ``common.env_util`` for clarity (affects ``make_vec_env`` and ``make_atari_env`` functions)

New Features:
^^^^^^^^^^^^^
- Allow custom actor/critic network architectures using ``net_arch=dict(qf=[400, 300], pi=[64, 64])`` for off-policy algorithms (SAC, TD3, DDPG)
- Added Hindsight Experience Replay ``HER``. (@megan-klaiber)
- ``VecNormalize`` now supports ``gym.spaces.Dict`` observation spaces
- Support logging videos to Tensorboard (@SwamyDev)
- Added ``share_features_extractor`` argument to ``SAC`` and ``TD3`` policies

Bug Fixes:
^^^^^^^^^^
- Fix GAE computation for on-policy algorithms (off-by one for the last value) (thanks @Wovchena)
- Fixed potential issue when loading a different environment
- Fix ignoring the exclude parameter when recording logs using json, csv or log as logging format (@SwamyDev)
- Make ``make_vec_env`` support the ``env_kwargs`` argument when using an env ID str (@ManifoldFR)
- Fix model creation initializing CUDA even when `device="cpu"` is provided
- Fix ``check_env`` not checking if the env has a Dict actionspace before calling ``_check_nan`` (@wmmc88)
- Update the check for spaces unsupported by Stable Baselines 3 to include checks on the action space (@wmmc88)
- Fixed features extractor bug for target network where the same net was shared instead
  of being separate. This bug affects ``SAC``, ``DDPG`` and ``TD3`` when using ``CnnPolicy`` (or custom features extractor)
- Fixed a bug when passing an environment when loading a saved model with a ``CnnPolicy``, the passed env was not wrapped properly
  (the bug was introduced when implementing ``HER`` so it should not be present in previous versions)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Improved typing coverage
- Improved error messages for unsupported spaces
- Added ``.vscode`` to the gitignore

Documentation:
^^^^^^^^^^^^^^
- Added first draft of migration guide
- Added intro to `imitation <https://github.com/HumanCompatibleAI/imitation>`_ library (@shwang)
- Enabled doc for ``CnnPolicies``
- Added advanced saving and loading example
- Added base doc for exporting models
- Added example for getting and setting model parameters


Pre-Release 0.9.0 (2020-10-03)
------------------------------

**Bug fixes, get/set parameters  and improved docs**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed ``device`` keyword argument of policies; use ``policy.to(device)`` instead. (@qxcv)
- Rename ``BaseClass.get_torch_variables`` -> ``BaseClass._get_torch_save_params`` and ``BaseClass.excluded_save_params`` -> ``BaseClass._excluded_save_params``
- Renamed saved items ``tensors`` to ``pytorch_variables`` for clarity
- ``make_atari_env``, ``make_vec_env`` and ``set_random_seed`` must be imported with (and not directly from ``stable_baselines3.common``):

.. code-block:: python

  from stable_baselines3.common.cmd_util import make_atari_env, make_vec_env
  from stable_baselines3.common.utils import set_random_seed


New Features:
^^^^^^^^^^^^^
- Added ``unwrap_vec_wrapper()`` to ``common.vec_env`` to extract ``VecEnvWrapper`` if needed
- Added ``StopTrainingOnMaxEpisodes`` to callback collection (@xicocaio)
- Added ``device`` keyword argument to ``BaseAlgorithm.load()`` (@liorcohen5)
- Callbacks have access to rollout collection locals as in SB2. (@PartiallyTyped)
- Added ``get_parameters`` and ``set_parameters`` for accessing/setting parameters of the agent
- Added actor/critic loss logging for TD3. (@mloo3)

Bug Fixes:
^^^^^^^^^^
- Added ``unwrap_vec_wrapper()`` to ``common.vec_env`` to extract ``VecEnvWrapper`` if needed
- Fixed a bug where the environment was reset twice when using ``evaluate_policy``
- Fix logging of ``clip_fraction`` in PPO (@diditforlulz273)
- Fixed a bug where cuda support was wrongly checked when passing the GPU index, e.g., ``device="cuda:0"`` (@liorcohen5)
- Fixed a bug when the random seed was not properly set on cuda when passing the GPU index

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Improve typing coverage of the ``VecEnv``
- Fix type annotation of ``make_vec_env`` (@ManifoldFR)
- Removed ``AlreadySteppingError`` and ``NotSteppingError`` that were not used
- Fixed typos in SAC and TD3
- Reorganized functions for clarity in ``BaseClass`` (save/load functions close to each other, private
  functions at top)
- Clarified docstrings on what is saved and loaded to/from files
- Simplified ``save_to_zip_file`` function by removing duplicate code
- Store library version along with the saved models
- DQN loss is now logged

Documentation:
^^^^^^^^^^^^^^
- Added ``StopTrainingOnMaxEpisodes`` details and example (@xicocaio)
- Updated custom policy section (added custom features extractor example)
- Re-enable ``sphinx_autodoc_typehints``
- Updated doc style for type hints and remove duplicated type hints



Pre-Release 0.8.0 (2020-08-03)
------------------------------

**DQN, DDPG, bug fixes and performance matching for Atari games**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- ``AtariWrapper`` and other Atari wrappers were updated to match SB2 ones
- ``save_replay_buffer`` now receives as argument the file path instead of the folder path (@tirafesi)
- Refactored ``Critic`` class for ``TD3`` and ``SAC``, it is now called ``ContinuousCritic``
  and has an additional parameter ``n_critics``
- ``SAC`` and ``TD3`` now accept an arbitrary number of critics (e.g. ``policy_kwargs=dict(n_critics=3)``)
  instead of only 2 previously

New Features:
^^^^^^^^^^^^^
- Added ``DQN`` Algorithm (@Artemis-Skade)
- Buffer dtype is now set according to action and observation spaces for ``ReplayBuffer``
- Added warning when allocation of a buffer may exceed the available memory of the system
  when ``psutil`` is available
- Saving models now automatically creates the necessary folders and raises appropriate warnings (@PartiallyTyped)
- Refactored opening paths for saving and loading to use strings, pathlib or io.BufferedIOBase (@PartiallyTyped)
- Added ``DDPG`` algorithm as a special case of ``TD3``.
- Introduced ``BaseModel`` abstract parent for ``BasePolicy``, which critics inherit from.

Bug Fixes:
^^^^^^^^^^
- Fixed a bug in the ``close()`` method of ``SubprocVecEnv``, causing wrappers further down in the wrapper stack to not be closed. (@NeoExtended)
- Fix target for updating q values in SAC: the entropy term was not conditioned by terminals states
- Use ``cloudpickle.load`` instead of ``pickle.load`` in ``CloudpickleWrapper``. (@shwang)
- Fixed a bug with orthogonal initialization when `bias=False` in custom policy (@rk37)
- Fixed approximate entropy calculation in PPO and A2C. (@andyshih12)
- Fixed DQN target network sharing features extractor with the main network.
- Fixed storing correct ``dones`` in on-policy algorithm rollout collection. (@andyshih12)
- Fixed number of filters in final convolutional layer in NatureCNN to match original implementation.

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Refactored off-policy algorithm to share the same ``.learn()`` method
- Split the ``collect_rollout()`` method for off-policy algorithms
- Added ``_on_step()`` for off-policy base class
- Optimized replay buffer size by removing the need of ``next_observations`` numpy array
- Optimized polyak updates (1.5-1.95 speedup) through inplace operations (@PartiallyTyped)
- Switch to ``black`` codestyle and added ``make format``, ``make check-codestyle`` and ``commit-checks``
- Ignored errors from newer pytype version
- Added a check when using ``gSDE``
- Removed codacy dependency from Dockerfile
- Added ``common.sb2_compat.RMSpropTFLike`` optimizer, which corresponds closer to the implementation of RMSprop from Tensorflow.

Documentation:
^^^^^^^^^^^^^^
- Updated notebook links
- Fixed a typo in the section of Enjoy a Trained Agent, in RL Baselines3 Zoo README. (@blurLake)
- Added Unity reacher to the projects page (@koulakis)
- Added PyBullet colab notebook
- Fixed typo in PPO example code (@joeljosephjin)
- Fixed typo in custom policy doc (@RaphaelWag)


Pre-Release 0.7.0 (2020-06-10)
------------------------------

**Hotfix for PPO/A2C + gSDE, internal refactoring and bug fixes**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- ``render()`` method of ``VecEnvs`` now only accept one argument: ``mode``
- Created new file common/torch_layers.py, similar to SB refactoring

  - Contains all PyTorch network layer definitions and features extractors: ``MlpExtractor``, ``create_mlp``, ``NatureCNN``

- Renamed ``BaseRLModel`` to ``BaseAlgorithm`` (along with offpolicy and onpolicy variants)
- Moved on-policy and off-policy base algorithms to ``common/on_policy_algorithm.py`` and ``common/off_policy_algorithm.py``, respectively.
- Moved ``PPOPolicy`` to ``ActorCriticPolicy`` in common/policies.py
- Moved ``PPO`` (algorithm class) into ``OnPolicyAlgorithm`` (``common/on_policy_algorithm.py``), to be shared with A2C
- Moved following functions from ``BaseAlgorithm``:

  - ``_load_from_file`` to ``load_from_zip_file`` (save_util.py)
  - ``_save_to_file_zip`` to ``save_to_zip_file`` (save_util.py)
  - ``safe_mean`` to ``safe_mean`` (utils.py)
  - ``check_env`` to ``check_for_correct_spaces`` (utils.py. Renamed to avoid confusion with environment checker tools)

- Moved static function ``_is_vectorized_observation`` from common/policies.py to common/utils.py under name ``is_vectorized_observation``.
- Removed ``{save,load}_running_average`` functions of ``VecNormalize`` in favor of ``load/save``.
- Removed ``use_gae`` parameter from ``RolloutBuffer.compute_returns_and_advantage``.

New Features:
^^^^^^^^^^^^^

Bug Fixes:
^^^^^^^^^^
- Fixed ``render()`` method for ``VecEnvs``
- Fixed ``seed()`` method for ``SubprocVecEnv``
- Fixed loading on GPU for testing when using gSDE and ``deterministic=False``
- Fixed ``register_policy`` to allow re-registering same policy for same sub-class (i.e. assign same value to same key).
- Fixed a bug where the gradient was passed when using ``gSDE`` with ``PPO``/``A2C``, this does not affect ``SAC``

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Re-enable unsafe ``fork`` start method in the tests (was causing a deadlock with tensorflow)
- Added a test for seeding ``SubprocVecEnv`` and rendering
- Fixed reference in NatureCNN (pointed to older version with different network architecture)
- Fixed comments saying "CxWxH" instead of "CxHxW" (same style as in torch docs / commonly used)
- Added bit further comments on register/getting policies ("MlpPolicy", "CnnPolicy").
- Renamed ``progress`` (value from 1 in start of training to 0 in end) to ``progress_remaining``.
- Added ``policies.py`` files for A2C/PPO, which define MlpPolicy/CnnPolicy (renamed ActorCriticPolicies).
- Added some missing tests for ``VecNormalize``, ``VecCheckNan`` and ``PPO``.

Documentation:
^^^^^^^^^^^^^^
- Added a paragraph on "MlpPolicy"/"CnnPolicy" and policy naming scheme under "Developer Guide"
- Fixed second-level listing in changelog


Pre-Release 0.6.0 (2020-06-01)
------------------------------

**Tensorboard support, refactored logger**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Remove State-Dependent Exploration (SDE) support for ``TD3``
- Methods were renamed in the logger:

  - ``logkv`` -> ``record``, ``writekvs`` -> ``write``, ``writeseq`` ->  ``write_sequence``,
  - ``logkvs`` -> ``record_dict``, ``dumpkvs`` -> ``dump``,
  - ``getkvs`` -> ``get_log_dict``, ``logkv_mean`` -> ``record_mean``,


New Features:
^^^^^^^^^^^^^
- Added env checker (Sync with Stable Baselines)
- Added ``VecCheckNan`` and ``VecVideoRecorder`` (Sync with Stable Baselines)
- Added determinism tests
- Added ``cmd_util`` and ``atari_wrappers``
- Added support for ``MultiDiscrete`` and ``MultiBinary`` observation spaces (@rolandgvc)
- Added ``MultiCategorical`` and ``Bernoulli`` distributions for PPO/A2C (@rolandgvc)
- Added support for logging to tensorboard (@rolandgvc)
- Added ``VectorizedActionNoise`` for continuous vectorized environments (@PartiallyTyped)
- Log evaluation in the ``EvalCallback`` using the logger

Bug Fixes:
^^^^^^^^^^
- Fixed a bug that prevented model trained on cpu to be loaded on gpu
- Fixed version number that had a new line included
- Fixed weird seg fault in docker image due to FakeImageEnv by reducing screen size
- Fixed ``sde_sample_freq`` that was not taken into account for SAC
- Pass logger module to ``BaseCallback`` otherwise they cannot write in the one used by the algorithms

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Renamed to Stable-Baseline3
- Added Dockerfile
- Sync ``VecEnvs`` with Stable-Baselines
- Update requirement: ``gym>=0.17``
- Added ``.readthedoc.yml`` file
- Added ``flake8`` and ``make lint`` command
- Added Github workflow
- Added warning when passing both ``train_freq`` and ``n_episodes_rollout`` to Off-Policy Algorithms

Documentation:
^^^^^^^^^^^^^^
- Added most documentation (adapted from Stable-Baselines)
- Added link to CONTRIBUTING.md in the README (@kinalmehta)
- Added gSDE project and update docstrings accordingly
- Fix ``TD3`` example code block


Pre-Release 0.5.0 (2020-05-05)
------------------------------

**CnnPolicy support for image observations, complete saving/loading for policies**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Previous loading of policy weights is broken and replace by the new saving/loading for policy

New Features:
^^^^^^^^^^^^^
- Added ``optimizer_class`` and ``optimizer_kwargs`` to ``policy_kwargs`` in order to easily
  customizer optimizers
- Complete independent save/load for policies
- Add ``CnnPolicy`` and ``VecTransposeImage`` to support images as input


Bug Fixes:
^^^^^^^^^^
- Fixed ``reset_num_timesteps`` behavior, so ``env.reset()`` is not called if ``reset_num_timesteps=True``
- Fixed ``squashed_output`` that was not pass to policy constructor for ``SAC`` and ``TD3`` (would result in scaled actions for unscaled action spaces)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Cleanup rollout return
- Added ``get_device`` util to manage PyTorch devices
- Added type hints to logger + use f-strings

Documentation:
^^^^^^^^^^^^^^


Pre-Release 0.4.0 (2020-02-14)
------------------------------

**Proper pre-processing, independent save/load for policies**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed CEMRL
- Model saved with previous versions cannot be loaded (because of the pre-preprocessing)

New Features:
^^^^^^^^^^^^^
- Add support for ``Discrete`` observation spaces
- Add saving/loading for policy weights, so the policy can be used without the model

Bug Fixes:
^^^^^^^^^^
- Fix type hint for activation functions

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Refactor handling of observation and action spaces
- Refactored features extraction to have proper preprocessing
- Refactored action distributions


Pre-Release 0.3.0 (2020-02-14)
------------------------------

**Bug fixes, sync with Stable-Baselines, code cleanup**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed default seed
- Bump dependencies (PyTorch and Gym)
- ``predict()`` now returns a tuple to match Stable-Baselines behavior

New Features:
^^^^^^^^^^^^^
- Better logging for ``SAC`` and ``PPO``

Bug Fixes:
^^^^^^^^^^
- Synced callbacks with Stable-Baselines
- Fixed colors in ``results_plotter``
- Fix entropy computation (now summed over action dim)

Others:
^^^^^^^
- SAC with SDE now sample only one matrix
- Added ``clip_mean`` parameter to SAC policy
- Buffers now return ``NamedTuple``
- More typing
- Add test for ``expln``
- Renamed ``learning_rate`` to ``lr_schedule``
- Add ``version.txt``
- Add more tests for distribution

Documentation:
^^^^^^^^^^^^^^
- Deactivated ``sphinx_autodoc_typehints`` extension


Pre-Release 0.2.0 (2020-02-14)
------------------------------

**Python 3.6+ required, type checking, callbacks, doc build**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Python 2 support was dropped, Stable Baselines3 now requires Python 3.6 or above
- Return type of ``evaluation.evaluate_policy()`` has been changed
- Refactored the replay buffer to avoid transformation between PyTorch and NumPy
- Created `OffPolicyRLModel` base class
- Remove deprecated JSON format for `Monitor`

New Features:
^^^^^^^^^^^^^
- Add ``seed()`` method to ``VecEnv`` class
- Add support for Callback (cf https://github.com/hill-a/stable-baselines/pull/644)
- Add methods for saving and loading replay buffer
- Add ``extend()`` method to the buffers
- Add ``get_vec_normalize_env()`` to ``BaseRLModel`` to retrieve ``VecNormalize`` wrapper when it exists
- Add ``results_plotter`` from Stable Baselines
- Improve ``predict()`` method to handle different type of observations (single, vectorized, ...)

Bug Fixes:
^^^^^^^^^^
- Fix loading model on CPU that were trained on GPU
- Fix ``reset_num_timesteps`` that was not used
- Fix entropy computation for squashed Gaussian (approximate it now)
- Fix seeding when using multiple environments (different seed per env)

Others:
^^^^^^^
- Add type check
- Converted all format string to f-strings
- Add test for ``OrnsteinUhlenbeckActionNoise``
- Add type aliases in ``common.type_aliases``

Documentation:
^^^^^^^^^^^^^^
- fix documentation build


Pre-Release 0.1.0 (2020-01-20)
------------------------------
**First Release: base algorithms and state-dependent exploration**

New Features:
^^^^^^^^^^^^^
- Initial release of A2C, CEM-RL, PPO, SAC and TD3, working only with ``Box`` input space
- State-Dependent Exploration (SDE) for A2C, PPO, SAC and TD3



Maintainers
-----------

Stable-Baselines3 is currently maintained by `Antonin Raffin`_ (aka `@araffin`_), `Ashley Hill`_ (aka @hill-a),
`Maximilian Ernestus`_ (aka @ernestum), `Adam Gleave`_ (`@AdamGleave`_), `Anssi Kanervisto`_ (aka `@Miffyli`_)
and `Quentin Gallouédec`_ (aka @qgallouedec).

.. _Ashley Hill: https://github.com/hill-a
.. _Antonin Raffin: https://araffin.github.io/
.. _Maximilian Ernestus: https://github.com/ernestum
.. _Adam Gleave: https://gleave.me/
.. _@araffin: https://github.com/araffin
.. _@AdamGleave: https://github.com/adamgleave
.. _Anssi Kanervisto: https://github.com/Miffyli
.. _@Miffyli: https://github.com/Miffyli
.. _Quentin Gallouédec: https://gallouedec.com/
.. _@qgallouedec: https://github.com/qgallouedec

.. _SB3-Contrib: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
.. _RL Zoo: https://github.com/DLR-RM/rl-baselines3-zoo

Contributors:
-------------
In random order...

Thanks to the maintainers of V2: @hill-a @enerijunior @AdamGleave @Miffyli

And all the contributors:
@taymuur @bjmuld @iambenzo @iandanforth @r7vme @brendenpetersen @huvar @abhiskk @JohannesAck
@EliasHasle @mrakgr @Bleyddyn @antoine-galataud @junhyeokahn @AdamGleave @keshaviyengar @tperol
@XMaster96 @kantneel @Pastafarianist @GerardMaggiolino @PatrickWalter214 @yutingsz @sc420 @Aaahh @billtubbs
@Miffyli @dwiel @miguelrass @qxcv @jaberkow @eavelardev @ruifeng96150 @pedrohbtp @srivatsankrishnan @evilsocket
@MarvineGothic @jdossgollin @stheid @SyllogismRXS @rusu24edward @jbulow @Antymon @seheevic @justinkterry @edbeeching
@flodorner @KuKuXia @NeoExtended @PartiallyTyped @mmcenta @richardwu @kinalmehta @rolandgvc @tkelestemur @mloo3
@tirafesi @blurLake @koulakis @joeljosephjin @shwang @rk37 @andyshih12 @RaphaelWag @xicocaio
@diditforlulz273 @liorcohen5 @ManifoldFR @mloo3 @SwamyDev @wmmc88 @megan-klaiber @thisray
@tfederico @hn2 @LucasAlegre @AptX395 @zampanteymedio @JadenTravnik @decodyng @ardabbour @lorenz-h @mschweizer @lorepieri8 @vwxyzjn
@ShangqunYu @PierreExeter @JacopoPan @ltbd78 @tom-doerr @Atlis @liusida @09tangriro @amy12xx @juancroldan
@benblack769 @bstee615 @c-rizz @skandermoalla @MihaiAnca13 @davidblom603 @ayeright @cyprienc
@wkirgsn @AechPro @CUN-bjy @batu @IljaAvadiev @timokau @kachayev @cleversonahum
@eleurent @ac-93 @cove9988 @theDebugger811 @hsuehch @Demetrio92 @thomasgubler @IperGiove @ScheiklP
@simoninithomas @armandpl @manuel-delverme @Gautam-J @gianlucadecola @buoyancy99 @caburu @xy9485
@Gregwar @ycheng517 @quantitative-technologies @bcollazo @git-thor @TibiGG @cool-RR @MWeltevrede
@carlosluis @arjun-kg @tlpss @JonathanKuelz @Gabo-Tor
@Melanol @qgallouedec @francescoluciano @jlp-ue @burakdmb @timothe-chaumont @honglu2875
@anand-bala @hughperkins @sidney-tio @AlexPasqua @dominicgkerr @Akhilez @Rocamonde @tobirohrer @ZikangXiong
@DavyMorgan @luizapozzobon @Bonifatius94 @theSquaredError @harveybellini @DavyMorgan @FieteO @jonasreiher @npit @WeberSamuel @troiganto
@lutogniew @lbergmann1 @lukashass @BertrandDecoster @pseudo-rnd-thoughts @stefanbschneider @kyle-he
