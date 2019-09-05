import gym

from torchy_baselines import TD3

def test_simple_run():
    env = gym.make("Pendulum-v0")
    model = TD3('MlpPolicy', env, policy_kwargs=dict(net_arch=[64, 64]), verbose=1)
    model.learn(total_timesteps=50000)
