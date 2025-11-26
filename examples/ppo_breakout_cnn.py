from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import ale_py        # the actual ALE backend
import shimmy        # registers ALE-py with Gymnasium

# 1. Create a vectorized Atari environment and stack frames
env = make_atari_env("ALE/Breakout-v5", n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

# 2. Instantiate PPO with a CNN policy
model = PPO(
    policy="CnnPolicy",
    env=env,
    verbose=1
)
# 3. Train the agent
model.learn(total_timesteps=200_000)

# 4. Save the trained model
model.save("ppo_breakout_cnn")

# 5. Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

# 6. (Optional) Run a few episodes and render
obs = env.reset()
for _ in range(10_000):
    action, _states = model.predict(obs)
    obs, rewards, dones, infos = env.step(action)
    env.render()
    if dones.any():
        obs = env.reset()
