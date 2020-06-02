# This file is here just to define MlpPolicy/CnnPolicy
# that work for A2C
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy, register_policy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy

register_policy("MlpPolicy", ActorCriticPolicy)
register_policy("CnnPolicy", ActorCriticCnnPolicy)
