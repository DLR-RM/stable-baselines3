.. _migration:

================================
Migrating from Stable-Baselines
================================


This is a guide to migrate from Stable-Baselines to Stable-Baselines3.

It also references the main changes.

.. warning::
	This section is still a Work In Progress (WIP) Things might be added in the future before 1.0 release.



Overview
========

Overall Stable-Baselines3 (SB3) keeps the high-level API of Stable-Baselines (SB2).
Most of the changes are to ensure more consistency and are internal ones.
Because of the backend change, from Tensorflow to PyTorch, the internal code is much much readable and easy to debug
at the cost of some speed (dynamic graph vs static graph., see `Issue #90 <https://github.com/DLR-RM/stable-baselines3/issues/90>`_)
However, the algorithms were extensively benchmarked (see `Issue #48 <https://github.com/DLR-RM/stable-baselines3/issues/48>`_  and `Issue #49 <https://github.com/DLR-RM/stable-baselines3/issues/49>`_)
so you should not expect performance drop when switching from SB2 to SB3.

Breaking Changes
================

- Dropped MPI support
- Dropped layer normalized policies (e.g. ``LnMlpPolicy``)
- Dropped parameter noise for DDPG and DQN
- PPO is now closer to the original implementation (no clipping of the value function by default), cf PPO section below
- orthogonal initialization is only used by A2C/PPO
- the features extractor (CNN extractor) is shared between policy and q-networks for DDPG/SAC/TD3 and only the policy loss used to update it (much faster)


TODO: change to deterministic predict for SAC/TD3

state api breaking changes and implementation differences (e.g. clip range ppo and renaming of parameters)

Moved Files
-----------

- ``bench/monitor.py`` -> ``common/monitor.py``
- ``logger.py`` -> ``common/logger.py``
- ``results_plotter.py`` -> ``common/results_plotter.py``


Parameters Change and Renaming
------------------------------

Policies
^^^^^^^^

- ``cnn_extractor`` -> ``feature_extractor``

A2C
^^^

PPO
^^^

- ``cliprange`` -> ``clip_range``
- ``cliprange_vf`` -> ``clip_range_vf``
- ``nminibatches`` -> ``batch_size``

.. warning::

	``nminibatches`` gave different batch size depending on the number of environments:  ``batch_size = (n_steps * n_envs) // nminibatches``


- ``clip_range_vf`` behavior for PPO is slightly different: Set it to ``None`` (default) to deactivate clipping (in SB2, you had to pass ``-1``, ``None`` meant to use ``clip_range`` for the clipping)


DQN
^^^

Only the vanilla DQN is implemented right now but extensions will follow (cf planned features).

DDPG
^^^^

DDPG now follows the same interface as SAC/TD3.
For state/reward normalization, you should use ``VecNormalize`` as for all other algorithms.

SAC/TD3
^^^^^^^

SAC/TD3 now accept any number of critics, e.g. ``policy_kwargs=dict(n_critics=3)``, instead of only two before.


New logger API
--------------

- Methods were renamed in the logger:

  - ``logkv`` -> ``record``, ``writekvs`` -> ``write``, ``writeseq`` ->  ``write_sequence``,
  - ``logkvs`` -> ``record_dict``, ``dumpkvs`` -> ``dump``,
  - ``getkvs`` -> ``get_log_dict``, ``logkv_mean`` -> ``record_mean``,


Internal Changes
----------------

Please read the :ref:`Developper Guide <developer>` section.


New Features
============

- much cleaner base code (and no more warnings =D )
- independent saving/loading/predict for policies
- A2C now supports Generalized Advantage Estimation (GAE) and advantage normalization (both are deactivated by default)
- generalized State-Dependent Exploration (gSDE) exploration is available for A2C/PPO/SAC. It allows to use RL directly on real robots (cf https://arxiv.org/abs/2005.05719)
- proper evaluation (using separate env) is included in the base class (using ``EvalCallback``),
  if you pass the environment as a string, you can pass ``create_eval_env=True`` to the algorithm constructor.
- better saving/loading: optimizers are now included in the saved parameters and there is two new methods ``save_replay_buffer`` and ``load_replay_buffer`` for the replay buffer when using off-policy algorithms (DQN/DDPG/SAC/TD3)
- you can pass ``optimizer_class`` and ``optimizer_kwargs`` to ``policy_kwargs`` in order to easily
  customize optimizers
- when using continuous actions
- seeding now works properly to have deterministic results
- replay buffer does not grow, allocate everything at build time (faster)


How to migrate?
===============

In most cases, replacing ``from stable_baselines`` by ``from stable_baselines3`` will be sufficient.
Some files were moved to the common folder (cf above) and could result to .

Planned Features
================

- Recurrent (LSTM) policies
- DQN extensions (the current implementation is a vanilla DQN)

cf `roadmap <https://github.com/DLR-RM/stable-baselines3/issues/1>`_
