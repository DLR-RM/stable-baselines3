import pytest

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

N_STEPS_TRAINING = 500
SEED = 0


@pytest.mark.parametrize("algo", [A2C, DQN, PPO, SAC, TD3])
def test_deterministic_training_common(algo):
    results = [[], []]
    rewards = [[], []]
    # Smaller network
    kwargs = {"policy_kwargs": dict(net_arch=[64])}
    env_id = "Pendulum-v1"
    if algo in [TD3, SAC]:
        kwargs.update({"action_noise": NormalActionNoise(0.0, 0.1), "learning_starts": 100, "train_freq": 4})
    else:
        if algo == DQN:
            env_id = "CartPole-v1"
            kwargs.update({"learning_starts": 100, "target_update_interval": 100})
        elif algo == PPO:
            kwargs.update({"n_steps": 64, "n_epochs": 4})

    for i in range(2):
        model = algo("MlpPolicy", env_id, seed=SEED, **kwargs)
        model.learn(N_STEPS_TRAINING)
        env = model.get_env()
        obs = env.reset()
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, _, _ = env.step(action)
            results[i].append(action)
            rewards[i].append(reward)
    assert sum(results[0]) == sum(results[1]), results
    assert sum(rewards[0]) == sum(rewards[1]), rewards
