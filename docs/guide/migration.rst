.. _migration:

================================
Migrating from Stable-Baselines
================================


This is a guide to migrate from Stable-Baselines to Stable-Baselines3.

It also references the main changes.

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

Dropped MPI

Implementation changes:

ppo closer to original implementation (no clip_range_vf by default)
a2c GAE enabled (not used by default)
SDE for SAC/A2C/PPO
ortho init only for A2C/PPO (including CNN)
CNN extractor shared for SAC/TD3 and only policy loss used to update it (much faster)
VecEnv is the default
TODO: change to deterministic predict for SAC/TD3


New Features
============

gSDE
replay buffer does not grow, allocate everything at build time
policy are independent (can be used to predict without a model)


How to migrate?
===============

Planned Features
================

cf `roadmap <https://github.com/DLR-RM/stable-baselines3/issues/1>`_
