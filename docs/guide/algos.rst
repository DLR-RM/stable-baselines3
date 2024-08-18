RL Algorithms
=============

This table displays the rl algorithms that are implemented in the Stable Baselines3 project,
along with some useful characteristics: support for discrete/continuous actions, multiprocessing.


===================  =========== ============ ================= =============== ================
Name                 ``Box``     ``Discrete`` ``MultiDiscrete`` ``MultiBinary`` Multi Processing
===================  =========== ============ ================= =============== ================
ARS [#f1]_           ✔️           ✔️            ❌                 ❌              ✔️
A2C                  ✔️           ✔️            ✔️                 ✔️               ✔️
DDPG                 ✔️           ❌            ❌                ❌               ✔️
DQN                  ❌           ✔️            ❌                ❌               ✔️
HER                  ✔️           ✔️            ❌                ❌               ✔️
PPO                  ✔️           ✔️            ✔️                 ✔️               ✔️
QR-DQN [#f1]_        ❌          ️ ✔️            ❌                ❌               ✔️
RecurrentPPO [#f1]_  ✔️           ✔️             ✔️                ✔️               ✔️
SAC                  ✔️           ❌            ❌                ❌               ✔️
TD3                  ✔️           ❌            ❌                ❌               ✔️
TQC [#f1]_           ✔️           ❌            ❌                ❌               ✔️
TRPO  [#f1]_         ✔️           ✔️            ✔️                 ✔️               ✔️
Maskable PPO [#f1]_  ❌           ✔️            ✔️                 ✔️               ✔️
===================  =========== ============ ================= =============== ================


.. [#f1] Implemented in `SB3 Contrib <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib>`_

.. note::
  ``Tuple`` observation spaces are not supported by any environment,
  however, single-level ``Dict`` spaces are (cf. :ref:`Examples <examples>`).


Actions ``gym.spaces``:

-  ``Box``: A N-dimensional box that contains every point in the action
   space.
-  ``Discrete``: A list of possible actions, where each timestep only
   one of the actions can be used.
-  ``MultiDiscrete``: A list of possible actions, where each timestep only one action of each discrete set can be used.
- ``MultiBinary``: A list of possible actions, where each timestep any of the actions can be used in any combination.


.. note::

  More algorithms (like QR-DQN or TQC) are implemented in our :ref:`contrib repo <sb3_contrib>`
  and in our :ref:`SBX (SB3 + Jax) repo <sbx>` (DroQ, CrossQ, ...).

.. note::

  Some logging values (like ``ep_rew_mean``, ``ep_len_mean``) are only available when using a ``Monitor`` wrapper
  See `Issue #339 <https://github.com/hill-a/stable-baselines/issues/339>`_ for more info.


.. note::

  When using off-policy algorithms, `Time Limits <https://arxiv.org/abs/1712.00378>`_ (aka timeouts) are handled
  properly (cf. `issue #284 <https://github.com/DLR-RM/stable-baselines3/issues/284>`_).
  You can revert to SB3 < 2.1.0 behavior by passing ``handle_timeout_termination=False``
  via the ``replay_buffer_kwargs`` argument.



Reproducibility
---------------

Completely reproducible results are not guaranteed across PyTorch releases or different platforms.
Furthermore, results need not be reproducible between CPU and GPU executions, even when using identical seeds.

In order to make computations deterministics, on your specific problem on one specific platform,
you need to pass a ``seed`` argument at the creation of a model.
If you pass an environment to the model using ``set_env()``, then you also need to seed the environment first.


Credit: part of the *Reproducibility* section comes from `PyTorch Documentation <https://pytorch.org/docs/stable/notes/randomness.html>`_
