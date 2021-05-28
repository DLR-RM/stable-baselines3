.. _callbacks:

Callbacks
=========

A callback is a set of functions that will be called at given stages of the training procedure.
You can use callbacks to access internal state of the RL model during training.
It allows one to do monitoring, auto saving, model manipulation, progress bars, ...


Custom Callback
---------------

To build a custom callback, you need to create a class that derives from ``BaseCallback``.
This will give you access to events (``_on_training_start``, ``_on_step``) and useful variables (like `self.model` for the RL model).


You can find two examples of custom callbacks in the documentation: one for saving the best model according to the training reward (see :ref:`Examples <examples>`), and one for logging additional values with Tensorboard (see :ref:`Tensorboard section <tensorboard>`).


.. code-block:: python

    from stable_baselines3.common.callbacks import BaseCallback


    class CustomCallback(BaseCallback):
        """
        A custom callback that derives from ``BaseCallback``.

        :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
        """
        def __init__(self, verbose=0):
            super(CustomCallback, self).__init__(verbose)
            # Those variables will be accessible in the callback
            # (they are defined in the base class)
            # The RL model
            # self.model = None  # type: BaseAlgorithm
            # An alias for self.model.get_env(), the environment used for training
            # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
            # Number of time the callback was called
            # self.n_calls = 0  # type: int
            # self.num_timesteps = 0  # type: int
            # local and global variables
            # self.locals = None  # type: Dict[str, Any]
            # self.globals = None  # type: Dict[str, Any]
            # The logger object, used to report things in the terminal
            # self.logger = None  # stable_baselines3.common.logger
            # # Sometimes, for event callback, it is useful
            # # to have access to the parent object
            # self.parent = None  # type: Optional[BaseCallback]

        def _on_training_start(self) -> None:
            """
            This method is called before the first rollout starts.
            """
            pass

        def _on_rollout_start(self) -> None:
            """
            A rollout is the collection of environment interaction
            using the current policy.
            This event is triggered before collecting new samples.
            """
            pass

        def _on_step(self) -> bool:
            """
            This method will be called by the model after each call to `env.step()`.

            For child callback (of an `EventCallback`), this will be called
            when the event is triggered.

            :return: (bool) If the callback returns False, training is aborted early.
            """
            return True

        def _on_rollout_end(self) -> None:
            """
            This event is triggered before updating the policy.
            """
            pass

        def _on_training_end(self) -> None:
            """
            This event is triggered before exiting the `learn()` method.
            """
            pass


.. note::
  ``self.num_timesteps`` corresponds to the total number of steps taken in the environment, i.e., it is the number of environments multiplied by the number of time ``env.step()`` was called

  For the other algorithms, ``self.num_timesteps`` is incremented by ``n_envs`` (number of environments) after each call to ``env.step()``


.. note::

  For off-policy algorithms like SAC, DDPG, TD3 or DQN, the notion of ``rollout`` corresponds to the steps taken in the environment between two updates.


.. _EventCallback:

Event Callback
--------------

Compared to Keras, Stable Baselines provides a second type of ``BaseCallback``, named ``EventCallback`` that is meant to trigger events. When an event is triggered, then a child callback is called.

As an example, :ref:`EvalCallback` is an ``EventCallback`` that will trigger its child callback when there is a new best model.
A child callback is for instance :ref:`StopTrainingOnRewardThreshold <StopTrainingCallback>` that stops the training if the mean reward achieved by the RL model is above a threshold.

.. note::

	We recommend to take a look at the source code of :ref:`EvalCallback` and :ref:`StopTrainingOnRewardThreshold <StopTrainingCallback>` to have a better overview of what can be achieved with this kind of callbacks.


.. code-block:: python

    class EventCallback(BaseCallback):
        """
        Base class for triggering callback on event.

        :param callback: (Optional[BaseCallback]) Callback that will be called
            when an event is triggered.
        :param verbose: (int)
        """
        def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
            super(EventCallback, self).__init__(verbose=verbose)
            self.callback = callback
            # Give access to the parent
            if callback is not None:
                self.callback.parent = self
        ...

        def _on_event(self) -> bool:
            if self.callback is not None:
                return self.callback()
            return True



Callback Collection
-------------------

Stable Baselines provides you with a set of common callbacks for:

- saving the model periodically (:ref:`CheckpointCallback`)
- evaluating the model periodically and saving the best one (:ref:`EvalCallback`)
- chaining callbacks (:ref:`CallbackList`)
- triggering callback on events (:ref:`EventCallback`, :ref:`EveryNTimesteps`)
- stopping the training early based on a reward threshold (:ref:`StopTrainingOnRewardThreshold <StopTrainingCallback>`)


.. _CheckpointCallback:

CheckpointCallback
^^^^^^^^^^^^^^^^^^

Callback for saving a model every ``save_freq`` calls to ``env.step()``, you must specify a log folder (``save_path``)
and optionally a prefix for the checkpoints (``rl_model`` by default).

.. warning::

  When using multiple environments, each call to  ``env.step()`` will effectively correspond to ``n_envs`` steps.
  If you want the ``save_freq`` to be similar when using different number of environments,
  you need to account for it using ``save_freq = max(save_freq // n_envs, 1)``.
  The same goes for the other callbacks.


.. code-block:: python

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                             name_prefix='rl_model')

    model = SAC('MlpPolicy', 'Pendulum-v0')
    model.learn(2000, callback=checkpoint_callback)


.. _EvalCallback:

EvalCallback
^^^^^^^^^^^^

Evaluate periodically the performance of an agent, using a separate test environment.
It will save the best model if ``best_model_save_path`` folder is specified and save the evaluations results in a numpy archive (``evaluations.npz``) if ``log_path`` folder is specified.


.. note::

	You can pass a child callback via the ``callback_on_new_best`` argument. It will be triggered each time there is a new best model.


.. warning::

  You need to make sure that ``eval_env`` is wrapped the same way as the training environment, for instance using the ``VecTransposeImage`` wrapper if you have a channel-last image as input.
  The ``EvalCallback`` class outputs a warning if it is not the case.


.. code-block:: python

    import gym

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback

    # Separate evaluation env
    eval_env = gym.make('Pendulum-v0')
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=500,
                                 deterministic=True, render=False)

    model = SAC('MlpPolicy', 'Pendulum-v0')
    model.learn(5000, callback=eval_callback)


.. _Callbacklist:

CallbackList
^^^^^^^^^^^^

Class for chaining callbacks, they will be called sequentially.
Alternatively, you can pass directly a list of callbacks to the ``learn()`` method, it will be converted automatically to a ``CallbackList``.


.. code-block:: python

    import gym

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/')
    # Separate evaluation env
    eval_env = gym.make('Pendulum-v0')
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model',
                                 log_path='./logs/results', eval_freq=500)
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    model = SAC('MlpPolicy', 'Pendulum-v0')
    # Equivalent to:
    # model.learn(5000, callback=[checkpoint_callback, eval_callback])
    model.learn(5000, callback=callback)


.. _StopTrainingCallback:

StopTrainingOnRewardThreshold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stop the training once a threshold in episodic reward (mean episode reward over the evaluations) has been reached (i.e., when the model is good enough).
It must be used with the :ref:`EvalCallback` and use the event triggered by a new best model.


.. code-block:: python

    import gym

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

    # Separate evaluation env
    eval_env = gym.make('Pendulum-v0')
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)

    model = SAC('MlpPolicy', 'Pendulum-v0', verbose=1)
    # Almost infinite number of timesteps, but the training will stop
    # early as soon as the reward threshold is reached
    model.learn(int(1e10), callback=eval_callback)


.. _EveryNTimesteps:

EveryNTimesteps
^^^^^^^^^^^^^^^

An :ref:`EventCallback` that will trigger its child callback every ``n_steps`` timesteps.


.. note::

	Because of the way ``PPO1`` and ``TRPO`` work (they rely on MPI), ``n_steps`` is a lower bound between two events.


.. code-block:: python

  import gym

  from stable_baselines3 import PPO
  from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps

  # this is equivalent to defining CheckpointCallback(save_freq=500)
  # checkpoint_callback will be triggered every 500 steps
  checkpoint_on_event = CheckpointCallback(save_freq=1, save_path='./logs/')
  event_callback = EveryNTimesteps(n_steps=500, callback=checkpoint_on_event)

  model = PPO('MlpPolicy', 'Pendulum-v0', verbose=1)

  model.learn(int(2e4), callback=event_callback)


.. _StopTrainingOnMaxEpisodes:

StopTrainingOnMaxEpisodes
^^^^^^^^^^^^^^^^^^^^^^^^^

Stop the training upon reaching the maximum number of episodes, regardless of the model's ``total_timesteps`` value.
Also, presumes that, for multiple environments, the desired behavior is that the agent trains on each env for ``max_episodes``
and in total for ``max_episodes * n_envs`` episodes.


.. note::
    For multiple environments, the agent will train for a total of ``max_episodes * n_envs`` episodes.
    However, it can't be guaranteed that this training will occur for an exact number of ``max_episodes`` per environment.
    Thus, there is an assumption that, on average, each environment ran for ``max_episodes``.


.. code-block:: python

    from stable_baselines3 import A2C
    from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

    # Stops training when the model reaches the maximum number of episodes
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=5, verbose=1)

    model = A2C('MlpPolicy', 'Pendulum-v0', verbose=1)
    # Almost infinite number of timesteps, but the training will stop
    # early as soon as the max number of episodes is reached
    model.learn(int(1e10), callback=callback_max_episodes)


.. automodule:: stable_baselines3.common.callbacks
  :members:
