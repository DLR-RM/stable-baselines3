.. _changelog:

Changelog
==========

Pre-Release 0.2.0a1 (WIP)
------------------------------

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Python 2 support was dropped, Torchy Baselines now requires Python 3.6 or above
- Return type of `evaluation.evaluate_policy()` has been changed
- Refactored the replay buffer to avoid transformation between PyTorch and NumPy
- Created `OffPolicyRLModel` base class

New Features:
^^^^^^^^^^^^^
- Add `seed()` method to `VecEnv` class
- Add support for Callback (cf https://github.com/hill-a/stable-baselines/pull/644)
- Add methods for saving and loading replay buffer
- Add `extend()` method to the buffers

Bug Fixes:
^^^^^^^^^^
- Fix loading model on CPU that were trained on GPU
- Fix `reset_num_timesteps` that was not used
- Fix entropy computation for squashed Gaussian (approximate it now)
- Fix seeding when using multiple environments (different seed per env)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Add type check
- Converted all format string to f-strings
- Add test for `OrnsteinUhlenbeckActionNoise`
- Add type aliases in `common.type_aliases`

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
- Initial release of A2C, CEM-RL, PPO, SAC and TD3, working only with `Box` input space
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

Thanks to @hill-a @enerijunior @AdamGleave @Miffyli
