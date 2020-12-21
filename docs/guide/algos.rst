RL Algorithms
=============

This table displays the rl algorithms that are implemented in the Stable Baselines3 project,
along with some useful characteristics: support for discrete/continuous actions, multiprocessing.


============ =========== ============ ================= =============== ================
Name         ``Box``     ``Discrete`` ``MultiDiscrete`` ``MultiBinary`` Multi Processing
============ =========== ============ ================= =============== ================
A2C          ✔️           ✔️            ✔️                 ✔️               ✔️
DDPG         ✔️          ❌            ❌                ❌              ❌
DQN          ❌           ✔️           ❌                ❌              ❌
HER          ✔️            ✔️           ❌                ❌              ❌
PPO          ✔️           ✔️            ✔️                 ✔️               ✔️
SAC          ✔️          ❌            ❌                ❌              ❌
TD3          ✔️          ❌            ❌                ❌              ❌
============ =========== ============ ================= =============== ================


.. note::
    Non-array spaces such as ``Dict`` or ``Tuple`` are not currently supported by any algorithm.

Actions ``gym.spaces``:

-  ``Box``: A N-dimensional box that contains every point in the action
   space.
-  ``Discrete``: A list of possible actions, where each timestep only
   one of the actions can be used.
-  ``MultiDiscrete``: A list of possible actions, where each timestep only one action of each discrete set can be used.
- ``MultiBinary``: A list of possible actions, where each timestep any of the actions can be used in any combination.


.. note::

  More algorithms (like QR-DQN or TQC) are implemented in our :ref:`contrib repo <sb3_contrib>`.

.. note::

  Some logging values (like ``ep_rew_mean``, ``ep_len_mean``) are only available when using a ``Monitor`` wrapper
  See `Issue #339 <https://github.com/hill-a/stable-baselines/issues/339>`_ for more info.


Reproducibility
---------------

Completely reproducible results are not guaranteed across Tensorflow releases or different platforms.
Furthermore, results need not be reproducible between CPU and GPU executions, even when using identical seeds.

In order to make computations deterministics, on your specific problem on one specific platform,
you need to pass a ``seed`` argument at the creation of a model.
If you pass an environment to the model using ``set_env()``, then you also need to seed the environment first.


Credit: part of the *Reproducibility* section comes from `PyTorch Documentation <https://pytorch.org/docs/stable/notes/randomness.html>`_
