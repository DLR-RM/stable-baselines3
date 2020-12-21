.. _sb3_contrib:

==================
SB3 Contrib
==================

We implement experimental features in a separate contrib repository:
`SB3-Contrib`_

This allows Stable-Baselines3 (SB3) to maintain a stable and compact core, while still
providing the latest features, like Truncated Quantile Critics (TQC) or
Quantile Regression DQN (QR-DQN).

Why create this repository?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Over the span of stable-baselines and stable-baselines3, the community
has been eager to contribute in form of better logging utilities,
environment wrappers, extended support (e.g. different action spaces)
and learning algorithms.

However sometimes these utilities were too niche to be considered for
stable-baselines or proved to be too difficult to integrate well into
the existing code without creating a mess. sb3-contrib aims to fix this by not
requiring the neatest code integration with existing code and not
setting limits on what is too niche: almost everything remotely useful
goes!
We hope this allows us to provide reliable implementations
following stable-baselines usual standards (consistent style, documentation, etc)
beyond the relatively small scope of utilities in the main repository.

Features
--------

See documentation for the full list of included features.

**RL Algorithms**:

- `Truncated Quantile Critics (TQC)`_
- `Quantile Regression DQN (QR-DQN)`_

**Gym Wrappers**:

- `Time Feature Wrapper`_

Documentation
-------------

Documentation is available online: https://sb3-contrib.readthedocs.io/

Installation
------------

To install Stable-Baselines3 contrib with pip, execute:

::

   pip install sb3-contrib

We recommend to use the ``master`` version of Stable Baselines3 and SB3-Contrib.

To install Stable Baselines3 ``master`` version:

::

   pip install git+https://github.com/DLR-RM/stable-baselines3

To install Stable Baselines3 contrib ``master`` version:

::

  pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib


Example
-------

SB3-Contrib follows the SB3 API and folder structure. So, if you are familiar with SB3,
using SB3-Contrib should be easy too.

Here is an example of training a Quantile Regression DQN (QR-DQN) agent on the CartPole environment.

.. code-block:: python

  from sb3_contrib import QRDQN

  policy_kwargs = dict(n_quantiles=50)
  model = QRDQN("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)
  model.learn(total_timesteps=10000, log_interval=4)
  model.save("qrdqn_cartpole")



.. _SB3-Contrib: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
.. _Truncated Quantile Critics (TQC): https://arxiv.org/abs/2005.04269
.. _Quantile Regression DQN (QR-DQN): https://arxiv.org/abs/1710.10044
.. _Time Feature Wrapper: https://arxiv.org/abs/1712.00378
