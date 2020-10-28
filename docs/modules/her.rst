.. _her:

.. automodule:: stable_baselines3.her


HER
====

`Hindsight Experience Replay (HER) <https://arxiv.org/abs/1707.01495>`_

HER is an algorithm that works with off-policy methods (DQN, SAC, TD3 and DDPG for example).
HER uses the fact that even if a desired goal was not achieved, other goal may have been achieved during a rollout.
It creates "virtual" transitions by relabeling transitions (changing the desired goal) from past episodes.



.. warning::

    HER requires the environment to inherits from `gym.GoalEnv <https://github.com/openai/gym/blob/3394e245727c1ae6851b504a50ba77c73cd4c65b/gym/core.py#L160>`_


.. warning::

  For performance reasons, the maximum number of steps per episodes must be specified.
  In most cases, it will be inferred if you specify ``max_episode_steps`` when registering the environment
  or if you use a ``gym.wrappers.TimeLimit`` (and ``env.spec`` is not None).
  Otherwise, you can directly pass ``max_episode_length`` to the model constructor


.. warning::

	``HER`` supports ``VecNormalize`` wrapper but only when ``online_sampling=True``


Notes
-----

- Original paper: https://arxiv.org/abs/1707.01495
- OpenAI paper: `Plappert et al. (2018)`_
- OpenAI blog post: https://openai.com/blog/ingredients-for-robotics-research/


.. _Plappert et al. (2018): https://arxiv.org/abs/1802.09464

Can I use?
----------

Please refer to the used model (DQN, SAC, TD3 or DDPG) for that section.

Example
-------

.. code-block:: python

    from stable_baselines3 import HER, DDPG, DQN, SAC, TD3
    from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
    from stable_baselines3.common.bit_flipping_env import BitFlippingEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper

    model_class = DQN  # works also with SAC, DDPG and TD3
    N_BITS = 15

    env = BitFlippingEnv(n_bits=N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)

    # Available strategies (cf paper): future, final, episode
    goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

    # If True the HER transitions will get sampled online
    online_sampling = True
    # Time limit for the episodes
    max_episode_length = N_BITS

    # Initialize the model
    model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy, online_sampling=online_sampling,
                            verbose=1, max_episode_length=max_episode_length)
    # Train the model
    model.learn(1000)

    model.save("./her_bit_env")
    model = HER.load('./her_bit_env', env=env)

    obs = env.reset()
    for _ in range(100):
        action, _ = model.model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        if done:
            obs = env.reset()


Results
-------

This implementation was tested on the `parking env <https://github.com/eleurent/highway-env>`_
using 3 seeds.

The complete learning curves are available in the `associated PR #120 <https://github.com/DLR-RM/stable-baselines3/pull/120>`_.



How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the `rl-zoo repo <https://github.com/DLR-RM/rl-baselines3-zoo>`_:

.. code-block:: bash

  git clone https://github.com/DLR-RM/rl-baselines3-zoo
  cd rl-baselines3-zoo/


Run the benchmark:

.. code-block:: bash

  python train.py --algo her --env parking-v0 --eval-episodes 10 --eval-freq 10000


Plot the results:

.. code-block:: bash

  python scripts/all_plots.py -a her -e parking-v0 -f logs/ --no-million


Parameters
----------

.. autoclass:: HER
  :members:

Goal Selection Strategies
-------------------------

.. autoclass:: GoalSelectionStrategy
  :members:
  :inherited-members:
    :undoc-members:


Obs Dict Wrapper
----------------

.. autoclass:: ObsDictWrapper
  :members:
  :inherited-members:
    :undoc-members:


HER Replay Buffer
-----------------

.. autoclass:: HerReplayBuffer
  :members:
  :inherited-members:
