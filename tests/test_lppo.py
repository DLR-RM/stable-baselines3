from stable_baselines3 import PPO
from stable_baselines3.common.buffers import MoRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import MoVecEnv, MoDummyVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.lppo.lppo import LPPO


def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value

    return func


from stable_baselines3.common.monitor import MoMonitor
from stable_baselines3.common.vec_env import MoVecMonitor


class EntropyScheduleCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.initial_ent_coef = 0.04
        self.final_ent_coef = 0.01
        self.total_timesteps = 25000000

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        current_ent_coef = self.initial_ent_coef - (self.initial_ent_coef - self.final_ent_coef) * progress
        self.model.ent_coef = current_ent_coef
        return True


from EthicalGatheringGame.MultiAgentEthicalGathering import MAEGG
from EthicalGatheringGame.presets import tiny

tiny['n_agents'] = 1
tiny["reward_mode"] = "vectorial"
tiny["objective_order"] = "ethical_first"
tiny["inequality_mode"] = "loss"
tiny["efficiency"] = [0.6]
tiny["donation_capacity"] = 15
tiny["survival_threshold"] = 30

def make_env():
    def _init():
        return MAEGG(**tiny)
    return _init
env = MoDummyVecEnv([make_env() for _ in range(5)],)
env = MoVecMonitor(env)

def linear_schedule_then_flat(initial_value):
    def func(progress_remaining):
        return max(progress_remaining * initial_value, 1)  # Linearly decrease

    return func

args = {
    "n_steps": 1000,
    "batch_size": 5000,
    "ent_coef": 0.04,
    "learning_rate": [linear_schedule(3e-5), linear_schedule(3e-4)],
    "rollout_buffer_class": MoRolloutBuffer,
    "rollout_buffer_kwargs": {},
    "beta_values": [2.0, 1.0],
    "eta_values": 5.0,
    "policy_kwargs": {
        "n_objectives": 2,
        "net_arch": dict(pi=[256, 128], vf=[256, 128])
    },
    "verbose": 1,
    "device": "cpu",
    "tensorboard_log": "runs",
    "clip_range_vf": 0.2,
    "gamma": 0.8,
    "normalize_advantage": True,
    "tolerance": 0.0,
    "recent_loses_len": 80,
    "n_epochs": 40
}


model = LPPO("MoMlpPolicy", env, 2, **args)

#model = PPO("MlpPolicy", env, **args)

model.learn(total_timesteps=10000, log_interval=1)
model.save("model")

#model.save("test")
env = MAEGG(**tiny)
env.toggleTrack(True)
env.toggleStash(True)


"""model = LPPO.load("test", env=env)

for ep in range(150):
    obs, _ = env.reset()
    for i in range(500):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, tr, tm, info = env.step(action.item())
        #env.render()
        if tr or tm:
            obs, _ = env.reset()

env.unwrapped.plot_results("median")"""