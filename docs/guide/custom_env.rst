.. _custom_env:

Using Custom Environments
==========================

To use the RL baselines with custom environments, they just need to follow the *gym* interface.
That is to say, your environment must implement the following methods (and inherits from OpenAI Gym Class):


.. note::
	If you are using images as input, the observation must be of type ``np.uint8`` and be contained in [0, 255]
	is normalized (dividing by 255 to have values in [0, 1]) when using CNN policies. Images can be either
	channel-first or channel-last.


.. note::

  Although SB3 supports both channel-last and channel-first images as input, we recommend using the channel-first convention when possible.
  Under the hood, when a channel-last image is passed, SB3 uses a ``VecTransposeImage`` wrapper to re-order the channels.



.. code-block:: python

  import gym
  from gym import spaces

  class CustomEnv(gym.Env):
      """Custom Environment that follows gym interface"""
      metadata = {"render.modes": ["human"]}

      def __init__(self, arg1, arg2, ...):
          super(CustomEnv, self).__init__()
          # Define action and observation space
          # They must be gym.spaces objects
          # Example when using discrete actions:
          self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
          # Example for using image as input (channel-first; channel-last also works):
          self.observation_space = spaces.Box(low=0, high=255,
                                              shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

      def step(self, action):
          ...
          return observation, reward, done, info
      def reset(self):
          ...
          return observation  # reward, done, info can't be included
      def render(self, mode="human"):
          ...
      def close (self):
          ...


Then you can define and train a RL agent with:

.. code-block:: python

  # Instantiate the env
  env = CustomEnv(arg1, ...)
  # Define and Train the agent
  model = A2C("CnnPolicy", env).learn(total_timesteps=1000)


To check that your environment follows the Gym interface that SB3 supports, please use:

.. code-block:: python

	from stable_baselines3.common.env_checker import check_env

	env = CustomEnv(arg1, ...)
	# It will check your custom environment and output additional warnings if needed
	check_env(env)

Gym also have its own `env checker <https://www.gymlibrary.ml/content/api/#checking-api-conformity>`_ but it checks a superset of what SB3 supports (SB3 does not support all Gym features).

We have created a `colab notebook <https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb>`_ for a concrete example on creating a custom environment along with an example of using it with Stable-Baselines3 interface.

Alternatively, you may look at OpenAI Gym `built-in environments <https://www.gymlibrary.ml/>`_. However, the readers are cautioned as per OpenAI Gym `official wiki <https://github.com/openai/gym/wiki/FAQ>`_, its advised not to customize their built-in environments. It is better to copy and create new ones if you need to modify them.

Optionally, you can also register the environment with gym, that will allow you to create the RL agent in one line (and use ``gym.make()`` to instantiate the env):

.. code-block:: python

	from gym.envs.registration import register
	# Example for the CartPole environment
	register(
	    # unique identifier for the env `name-version`
	    id="CartPole-v1",
	    # path to the class for creating the env
	    # Note: entry_point also accept a class as input (and not only a string)
	    entry_point="gym.envs.classic_control:CartPoleEnv",
	    # Max number of steps per episode, using a `TimeLimitWrapper`
	    max_episode_steps=500,
	)



In the project, for testing purposes, we use a custom environment named ``IdentityEnv``
defined `in this file <https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/envs/identity_env.py>`_.
An example of how to use it can be found `here <https://github.com/DLR-RM/stable-baselines3/blob/master/tests/test_identity.py>`_.
