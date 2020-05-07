.. _custom_env:

Using Custom Environments
==========================

To use the rl baselines with custom environments, they just need to follow the *gym* interface.
That is to say, your environment must implement the following methods (and inherits from OpenAI Gym Class):


.. note::
	If you are using images as input, the input values must be in [0, 255] as the observation
	is normalized (dividing by 255 to have values in [0, 1]) when using CNN policies.



.. code-block:: python

  import gym
  from gym import spaces

  class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, arg1, arg2, ...):
      super(CustomEnv, self).__init__()
      # Define action and observation space
      # They must be gym.spaces objects
      # Example when using discrete actions:
      self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
      # Example for using image as input:
      self.observation_space = spaces.Box(low=0, high=255,
                                          shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

    def step(self, action):
      ...
      return observation, reward, done, info
    def reset(self):
      ...
      return observation  # reward, done, info can't be included
    def render(self, mode='human'):
      ...
    def close (self):
      ...


Then you can define and train a RL agent with:

.. code-block:: python

  # Instantiate the env
  env = CustomEnv(arg1, ...)
  # Define and Train the agent
  model = A2C('CnnPolicy', env).learn(total_timesteps=1000)


To check that your environment follows the gym interface, please use:

.. code-block:: python

	from stable_baselines3.common.env_checker import check_env

	env = CustomEnv(arg1, ...)
	# It will check your custom environment and output additional warnings if needed
	check_env(env)



We have created a `colab notebook <https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb>`_ for
a concrete example of creating a custom environment.

You can also find a `complete guide online <https://github.com/openai/gym/blob/master/docs/creating-environments.md>`_
on creating a custom Gym environment.


Optionally, you can also register the environment with gym,
that will allow you to create the RL agent in one line (and use ``gym.make()`` to instantiate the env).


In the project, for testing purposes, we use a custom environment named ``IdentityEnv``
defined `in this file <https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/identity_env.py>`_.
An example of how to use it can be found `here <https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py>`_.
