.. _changelog:

Changelog
==========

Pre-Release 0.9.0a0 (WIP)
------------------------------

Breaking Changes:
^^^^^^^^^^^^^^^^^

New Features:
^^^^^^^^^^^^^
- Added ``unwrap_vec_wrapper()`` to ``common.vec_env`` to extract ``VecEnvWrapper`` if needed

Bug Fixes:
^^^^^^^^^^
- Fixed a bug where the environment was reset twice when using ``evaluate_policy``

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Improve typing coverage of the ``VecEnv``
- Removed ``AlreadySteppingError`` and ``NotSteppingError`` that were not used

Documentation:
^^^^^^^^^^^^^^

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
- Fixed DQN target network sharing feature extractor with the main network.
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

  - Contains all PyTorch network layer definitions and feature extractors: ``MlpExtractor``, ``create_mlp``, ``NatureCNN``

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
`Maximilian Ernestus`_ (aka @erniejunior), `Adam Gleave`_ (`@AdamGleave`_) and `Anssi Kanervisto`_ (aka `@Miffyli`_).

.. _Ashley Hill: https://github.com/hill-a
.. _Antonin Raffin: https://araffin.github.io/
.. _Maximilian Ernestus: https://github.com/erniejunior
.. _Adam Gleave: https://gleave.me/
.. _@araffin: https://github.com/araffin
.. _@AdamGleave: https://github.com/adamgleave
.. _Anssi Kanervisto: https://github.com/Miffyli
.. _@Miffyli: https://github.com/Miffyli



Contributors:
-------------
In random order...

Thanks to the maintainers of V2: @hill-a @enerijunior @AdamGleave @Miffyli

And all the contributors:
@bjmuld @iambenzo @iandanforth @r7vme @brendenpetersen @huvar @abhiskk @JohannesAck
@EliasHasle @mrakgr @Bleyddyn @antoine-galataud @junhyeokahn @AdamGleave @keshaviyengar @tperol
@XMaster96 @kantneel @Pastafarianist @GerardMaggiolino @PatrickWalter214 @yutingsz @sc420 @Aaahh @billtubbs
@Miffyli @dwiel @miguelrass @qxcv @jaberkow @eavelardev @ruifeng96150 @pedrohbtp @srivatsankrishnan @evilsocket
@MarvineGothic @jdossgollin @SyllogismRXS @rusu24edward @jbulow @Antymon @seheevic @justinkterry @edbeeching
@flodorner @KuKuXia @NeoExtended @PartiallyTyped @mmcenta @richardwu @kinalmehta @rolandgvc @tkelestemur @mloo3
@tirafesi @blurLake @koulakis @joeljosephjin @shwang @rk37 @andyshih12 @RaphaelWag
