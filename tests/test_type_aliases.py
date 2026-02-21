from dataclasses import dataclass

import torch as th

from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)


@dataclass(frozen=True)
class CustomReplayBufferSamples(ReplayBufferSamples):
    metadata: str = ""


@dataclass(frozen=True)
class CustomRolloutBufferSamples(RolloutBufferSamples):
    metadata: str = ""


@dataclass(frozen=True)
class CustomDictReplayBufferSamples(DictReplayBufferSamples):
    metadata: str = ""


@dataclass(frozen=True)
class CustomDictRolloutBufferSamples(DictRolloutBufferSamples):
    metadata: str = ""


def test_samples_support_subclassing():
    replay = CustomReplayBufferSamples(
        observations=th.zeros(1, 2),
        actions=th.zeros(1, 3),
        next_observations=th.ones(1, 2),
        dones=th.zeros(1, 1, dtype=th.bool),
        rewards=th.ones(1),
        discounts=th.ones(1),
        metadata="replay",
    )
    rollout = CustomRolloutBufferSamples(
        observations=th.zeros(1, 2),
        actions=th.zeros(1, 3),
        old_values=th.zeros(1),
        old_log_prob=th.zeros(1),
        advantages=th.zeros(1),
        returns=th.zeros(1),
        metadata="rollout",
    )
    dict_replay = CustomDictReplayBufferSamples(
        observations={"obs": th.zeros(1, 2)},
        actions=th.zeros(1, 3),
        next_observations={"obs": th.ones(1, 2)},
        dones=th.zeros(1, 1, dtype=th.bool),
        rewards=th.ones(1),
        discounts=th.ones(1),
        metadata="dict_replay",
    )
    dict_rollout = CustomDictRolloutBufferSamples(
        observations={"obs": th.zeros(1, 2)},
        actions=th.zeros(1, 3),
        old_values=th.zeros(1),
        old_log_prob=th.zeros(1),
        advantages=th.zeros(1),
        returns=th.ones(1),
        metadata="dict_rollout",
    )

    assert isinstance(replay, ReplayBufferSamples)
    assert isinstance(rollout, RolloutBufferSamples)
    assert isinstance(dict_replay, DictReplayBufferSamples)
    assert isinstance(dict_rollout, DictRolloutBufferSamples)

    assert hasattr(replay, "_fields")
    assert hasattr(replay, "_asdict")
    assert hasattr(replay, "_replace")
    assert len(replay._fields) == len(replay) == 7
    assert len(rollout._fields) == len(rollout) == 7
    assert len(dict_replay._fields) == len(dict_replay) == 7
    assert len(dict_rollout._fields) == len(dict_rollout) == 7

    assert next(iter(replay)) is replay.observations
    assert dict(replay._asdict())["metadata"] == "replay"

    replaced_replay = replay._replace(metadata="replacement")
    assert replaced_replay.metadata == "replacement"
    assert type(replaced_replay) is type(replay)
