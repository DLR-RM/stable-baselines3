from frozenlake_plus_env import FrozenLakePlus

env = FrozenLakePlus(dynamic_slippery=True, slippery_change_freq=3)
obs, info = env.reset()

done = False
step = 0  # Initialize step counter
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    step += 1  # Increment step counter
    print(f"Step: {step}, Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    done = terminated or truncated
env.close()
