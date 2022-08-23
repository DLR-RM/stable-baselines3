from stable_baselines3 import PPO
from stable_baselines3.common import buffers
import torch as th
from unittest import mock
from gym import spaces
import numpy as np


def test_batch_size_1_not_nan():
    """
    Batch size 1 can give nan with normalized advantage because
    torch.std(some_length_1_tensor) == nan
    """
    self = mock.Mock()
    self.n_epochs = 1
    obs_space = spaces.Discrete(3)
    action_space = spaces.Discrete(2)
    rollout_buffer = buffers.RolloutBuffer(
        buffer_size=2, observation_space=obs_space, action_space=action_space)
    for _ in range(2):
        rollout_buffer.add(
            obs=np.zeros(1), action=np.zeros(1), reward=np.zeros(1), episode_start=np.zeros(1),
            value=th.zeros(1), log_prob=th.zeros(1))
    rollout_buffer.compute_returns_and_advantage(last_values=th.zeros(1), dones=np.zeros(1))
    self.rollout_buffer = rollout_buffer
    self.batch_size = 1
    self.action_space = action_space
    self.clip_range_vf = None
    self.ent_coef = 0.0
    self.vf_coef = 0.0
    self.normalize_advantage = True
    self.target_kl = None
    self.max_grad_norm = 0.5
    self._n_updates = 0
    self.clip_range.return_value = 0.2

    self.policy = mock.Mock()
    self.policy.parameters.return_value = []
    self.policy.evaluate_actions.return_value = (
        th.zeros(1, requires_grad=True), th.zeros(1, requires_grad=True), th.zeros(1)
    )
    self.policy.log_std = th.zeros(1)

    loss = None

    def record(key, value, **kwargs):
        nonlocal loss
        if key == 'train/loss':
            loss = value
    self.logger.record = record

    PPO.train(self=self)
    assert loss is not None
    assert loss == loss  # check not nan (since nan does not equal nan)
