.. _changelog:

Changelog
==========

Pre-Release 0.4.0a0 (WIP)
------------------------------


Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed CEMRL

New Features:
^^^^^^^^^^^^^

Bug Fixes:
^^^^^^^^^^

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Refactor handling of observation and action spaces

Documentation:
^^^^^^^^^^^^^^


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

Deprecations:
^^^^^^^^^^^^^

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
- Python 2 support was dropped, Torchy Baselines now requires Python 3.6 or above
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

Deprecations:
^^^^^^^^^^^^^

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

Breaking Changes:
^^^^^^^^^^^^^^^^^

New Features:
^^^^^^^^^^^^^
- Initial release of A2C, CEM-RL, PPO, SAC and TD3, working only with ``Box`` input space
- State-Dependent Exploration (SDE) for A2C, PPO, SAC and TD3

Bug Fixes:
^^^^^^^^^^

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^

Documentation:
^^^^^^^^^^^^^^


Maintainers
-----------

Torchy-Baselines is currently maintained by `Antonin Raffin`_ (aka `@araffin`_).

.. _Antonin Raffin: https://araffin.github.io/
.. _@araffin: https://github.com/araffin



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
@flodorner @KuKuXia @NeoExtended @solliet @mmcenta @richardwu
