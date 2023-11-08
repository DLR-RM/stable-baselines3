.. _migration:

================================
Migrating from Stable-Baselines
================================


This is a guide to migrate from Stable-Baselines (SB2) to Stable-Baselines3 (SB3).

It also references the main changes.


Overview
========

Overall Stable-Baselines3 (SB3) keeps the high-level API of Stable-Baselines (SB2).
Most of the changes are to ensure more consistency and are internal ones.
Because of the backend change, from Tensorflow to PyTorch, the internal code is much more readable and easy to debug
at the cost of some speed (dynamic graph vs static graph., see `Issue #90 <https://github.com/DLR-RM/stable-baselines3/issues/90>`_)
However, the algorithms were extensively benchmarked on Atari games and continuous control PyBullet envs
(see `Issue #48 <https://github.com/DLR-RM/stable-baselines3/issues/48>`_  and `Issue #49 <https://github.com/DLR-RM/stable-baselines3/issues/49>`_)
so you should not expect performance drop when switching from SB2 to SB3.


How to migrate?
===============

In most cases, replacing ``from stable_baselines`` by ``from stable_baselines3`` will be sufficient.
Some files were moved to the common folder (cf below) and could result to import errors.
Some algorithms were removed because of their complexity to improve the maintainability of the project.
We recommend reading this guide carefully to understand all the changes that were made.
You can also take a look at the `rl-zoo3 <https://github.com/DLR-RM/rl-baselines3-zoo>`_ and compare the imports
to the `rl-zoo <https://github.com/araffin/rl-baselines-zoo>`_ of SB2 to have a concrete example of successful migration.


.. note::

  If you experience massive slow-down switching to PyTorch, you may need to play with the number of threads used,
  using ``torch.set_num_threads(1)`` or ``OMP_NUM_THREADS=1``, see `issue #122 <https://github.com/DLR-RM/stable-baselines3/issues/122>`_
  and `issue #90 <https://github.com/DLR-RM/stable-baselines3/issues/90>`_.


Breaking Changes
================


- SB3 requires python 3.7+ (instead of python 3.5+ for SB2)
- Dropped MPI support
- Dropped layer normalized policies (``MlpLnLstmPolicy``, ``CnnLnLstmPolicy``)
- LSTM policies (```MlpLstmPolicy```, ```CnnLstmPolicy```) are not supported for the time being
  (see `PR #53 <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/53>`_ for a recurrent PPO implementation)
- Dropped parameter noise for DDPG and DQN
- PPO is now closer to the original implementation (no clipping of the value function by default), cf PPO section below
- Orthogonal initialization is only used by A2C/PPO
- The features extractor (CNN extractor) is shared between policy and q-networks for DDPG/SAC/TD3 and only the policy loss used to update it (much faster)
- Tensorboard legacy logging was dropped in favor of having one logger for the terminal and Tensorboard (cf :ref:`Tensorboard integration <tensorboard>`)
- We dropped ACKTR/ACER support because of their complexity compared to simpler alternatives (PPO, SAC, TD3) performing as good.
- We dropped GAIL support as we are focusing on model-free RL only, you can however take a look at the :ref:`imitation project <imitation>` which implements
  GAIL and other imitation learning algorithms on top of SB3.
- ``action_probability`` is currently not implemented in the base class
- ``pretrain()`` method for behavior cloning was removed (see `issue #27 <https://github.com/DLR-RM/stable-baselines3/issues/27>`_)

You can take a look at the `issue about SB3 implementation design <https://github.com/hill-a/stable-baselines/issues/576>`_ for more details.


Moved Files
-----------

- ``bench/monitor.py`` -> ``common/monitor.py``
- ``logger.py`` -> ``common/logger.py``
- ``results_plotter.py`` -> ``common/results_plotter.py``
- ``common/cmd_util.py`` -> ``common/env_util.py``

Utility functions are no longer exported from ``common`` module, you should import them with their absolute path, e.g.:

.. code-block:: python

  from stable_baselines3.common.env_util import make_atari_env, make_vec_env
  from stable_baselines3.common.utils import set_random_seed

instead of ``from stable_baselines3.common import make_atari_env``



Changes and renaming in parameters
----------------------------------

Base-class (all algorithms)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``load_parameters`` -> ``set_parameters``

  - ``get/set_parameters`` return a dictionary mapping object names
    to their respective PyTorch tensors and other objects representing
    their parameters, instead of simpler mapping of parameter name to
    a NumPy array. These functions also return PyTorch tensors rather
    than NumPy arrays.


Policies
^^^^^^^^

- ``cnn_extractor`` -> ``features_extractor``, as ``features_extractor`` in now used with ``MlpPolicy`` too

A2C
^^^

- ``epsilon`` -> ``rms_prop_eps``
- ``lr_schedule`` is part of ``learning_rate`` (it can be a callable).
- ``alpha``, ``momentum`` are modifiable through ``policy_kwargs`` key ``optimizer_kwargs``.

.. warning::

	PyTorch implementation of RMSprop `differs from Tensorflow's <https://github.com/pytorch/pytorch/issues/23796>`_,
	which leads to `different and potentially more unstable results <https://github.com/DLR-RM/stable-baselines3/pull/110#issuecomment-663255241>`_.
	Use ``stable_baselines3.common.sb2_compat.rmsprop_tf_like.RMSpropTFLike`` optimizer to match the results
	with TensorFlow's implementation. This can be done through ``policy_kwargs``: ``A2C(policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)))``


PPO
^^^

- ``cliprange`` -> ``clip_range``
- ``cliprange_vf`` -> ``clip_range_vf``
- ``nminibatches`` -> ``batch_size``

.. warning::

	``nminibatches`` gave different batch size depending on the number of environments:  ``batch_size = (n_steps * n_envs) // nminibatches``


- ``clip_range_vf`` behavior for PPO is slightly different: Set it to ``None`` (default) to deactivate clipping (in SB2, you had to pass ``-1``, ``None`` meant to use ``clip_range`` for the clipping)
- ``lam`` -> ``gae_lambda``
- ``noptepochs`` -> ``n_epochs``

PPO default hyperparameters are the one tuned for continuous control environment.
We recommend taking a look at the :ref:`RL Zoo <rl_zoo>` for hyperparameters tuned for Atari games.


DQN
^^^

Only the vanilla DQN is implemented right now but extensions will follow.
Default hyperparameters are taken from the Nature paper, except for the optimizer and learning rate that were taken from Stable Baselines defaults.

DDPG
^^^^

DDPG now follows the same interface as SAC/TD3.
For state/reward normalization, you should use ``VecNormalize`` as for all other algorithms.

SAC/TD3
^^^^^^^

SAC/TD3 now accept any number of critics, e.g. ``policy_kwargs=dict(n_critics=3)``, instead of only two before.


.. note::

	SAC/TD3 default hyperparameters (including network architecture) now match the ones from the original papers.
	DDPG is using TD3 defaults.


SAC
^^^

SAC implementation matches the latest version of the original implementation: it uses two Q function networks and two target Q function networks
instead of two Q function networks and one Value function network (SB2 implementation, first version of the original implementation).
Despite this change, no change in performance should be expected.

.. note::

	SAC ``predict()`` method has now ``deterministic=False`` by default for consistency.
	To match SB2 behavior, you need to explicitly pass ``deterministic=True``


HER
^^^

The ``HER`` implementation now only supports online sampling of the new goals. This is done in a vectorized version.
The goal selection strategy ``RANDOM`` is no longer supported.


New logger API
--------------

- Methods were renamed in the logger:

  - ``logkv`` -> ``record``, ``writekvs`` -> ``write``, ``writeseq`` ->  ``write_sequence``,
  - ``logkvs`` -> ``record_dict``, ``dumpkvs`` -> ``dump``,
  - ``getkvs`` -> ``get_log_dict``, ``logkv_mean`` -> ``record_mean``,


Internal Changes
----------------

Please read the :ref:`Developer Guide <developer>` section.


New Features (SB3 vs SB2)
=========================

- Much cleaner and consistent base code (and no more warnings =D!) and static type checks
- Independent saving/loading/predict for policies
- A2C now supports Generalized Advantage Estimation (GAE) and advantage normalization (both are deactivated by default)
- Generalized State-Dependent Exploration (gSDE) exploration is available for A2C/PPO/SAC. It allows using RL directly on real robots (cf https://arxiv.org/abs/2005.05719)
- Better saving/loading: optimizers are now included in the saved parameters and there are two new methods ``save_replay_buffer`` and ``load_replay_buffer`` for the replay buffer when using off-policy algorithms (DQN/DDPG/SAC/TD3)
- You can pass ``optimizer_class`` and ``optimizer_kwargs`` to ``policy_kwargs`` in order to easily
  customize optimizers
- Seeding now works properly to have deterministic results
- Replay buffer does not grow, allocate everything at build time (faster)
- We added a memory efficient replay buffer variant (pass ``optimize_memory_usage=True`` to the constructor), it reduces drastically the memory used especially when using images
- You can specify an arbitrary number of critics for SAC/TD3 (e.g. ``policy_kwargs=dict(n_critics=3)``)
