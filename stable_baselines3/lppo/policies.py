from torch import nn
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy

# TODO: Create CNN and RNN versions of the policy

class MoActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, n_objectives, **kwargs):
        self.n_objectives = n_objectives
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, self.n_objectives)

MoMlpPolicy = MoActorCriticPolicy