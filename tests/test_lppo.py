from stable_baselines3.common.buffers import MoRolloutBuffer
from stable_baselines3.common.vec_env import MoVecEnv, MoDummyVecEnv
from stable_baselines3.lppo.lppo import LPPO


def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value

    return func


from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, MoEvalCallback


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
tiny["inequality_mode"] = "tie"
tiny["efficiency"] = 0.6
tiny["donation_capacity"] = 15
tiny["survival_threshold"] = 30

env = MoDummyVecEnv([lambda: MAEGG(**tiny) for _ in range(5)], n_objectives=2)

args = {
    "n_steps": 500,
    "batch_size": 2500,
    "ent_coef": 0.04,
    "learning_rate": linear_schedule(3e-2),
    "rollout_buffer_class": MoRolloutBuffer,
    "rollout_buffer_kwargs": {},
    "beta_values": [2.0, 1.0],
    "eta_values": 5.0,
    "policy_kwargs": {"n_objectives": 2},
    "verbose": 1,
    "device": "cpu",
    "tensorboard_log": "runs",
    "n_epochs": 25,
    "clip_range_vf": 0.2,
    "gamma": 0.8,
    "normalize_advantage": False,
}


model = LPPO("MoMlpPolicy", env, 2, **args)
model.learn(total_timesteps=25000000, callback=[
    EntropyScheduleCallback(),
    MoEvalCallback(env, n_objectives=2, deterministic=False, n_eval_episodes=200, eval_freq=100000)
                                                ], log_interval=50)
model.save("test")