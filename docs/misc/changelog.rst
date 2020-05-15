.. _changelog:

Changelog
==========

Pre-Release 0.6.0a8 (WIP)
------------------------------


Breaking Changes:
^^^^^^^^^^^^^^^^^
- Remove State-Dependent Exploration (SDE) support for ``TD3``

New Features:
^^^^^^^^^^^^^
- Added env checker (Sync with Stable Baselines)
- Added ``VecCheckNan`` and ``VecVideoRecorder`` (Sync with Stable Baselines)
- Added determinism tests
- Added ``cmd_utils`` and ``atari_wrappers``

Bug Fixes:
^^^^^^^^^^
- Fixed a bug that prevented model trained on cpu to be loaded on gpu
- Fixed version number that had a new line included
- Fixed weird seg fault in docker image due to FakeImageEnv by reducing screen size

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

Documentation:
^^^^^^^^^^^^^^
- Added most documentation (adapted from Stable-Baselines)
- Added link to CONTRIBUTING.md in the README (@kinalmehta)
- Added gSDE project and update docstrings accordingly


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
@flodorner @KuKuXia @NeoExtended @solliet @mmcenta @richardwu @kinalmehta
