import os
import sys
import time
from typing import Sequence
from unittest import mock

import gym
import numpy as np
import pytest
import torch as th
from matplotlib import pyplot as plt
from pandas.errors import EmptyDataError

from stable_baselines3 import A2C, DQN
from stable_baselines3.common.logger import (
    DEBUG,
    INFO,
    CSVOutputFormat,
    Figure,
    FormatUnsupportedError,
    HParam,
    HumanOutputFormat,
    Image,
    Logger,
    TensorBoardOutputFormat,
    Video,
    configure,
    make_output_format,
    read_csv,
    read_json,
)

KEY_VALUES = {
    "test": 1,
    "b": -3.14,
    "8": 9.9,
    "l": [1, 2],
    "a": np.array([1, 2, 3]),
    "f": np.array(1),
    "g": np.array([[[1]]]),
    "h": 'this ", ;is a \n tes:,t',
}

KEY_EXCLUDED = {}
for key in KEY_VALUES.keys():
    KEY_EXCLUDED[key] = None


class LogContent:
    """
    A simple wrapper class to provide a common interface to check content for emptiness and report the log format
    """

    def __init__(self, _format: str, lines: Sequence):
        self.format = _format
        self.lines = lines

    @property
    def empty(self):
        return len(self.lines) == 0

    def __repr__(self):
        return f"LogContent(_format={self.format}, lines={self.lines})"


@pytest.fixture
def read_log(tmp_path, capsys):
    def read_fn(_format):
        if _format == "csv":
            try:
                df = read_csv(tmp_path / "progress.csv")
            except EmptyDataError:
                return LogContent(_format, [])
            return LogContent(_format, [r for _, r in df.iterrows() if not r.empty])
        elif _format == "json":
            try:
                df = read_json(tmp_path / "progress.json")
            except EmptyDataError:
                return LogContent(_format, [])
            return LogContent(_format, [r for _, r in df.iterrows() if not r.empty])
        elif _format == "stdout":
            captured = capsys.readouterr()
            return LogContent(_format, captured.out.splitlines())
        elif _format == "log":
            return LogContent(_format, (tmp_path / "log.txt").read_text().splitlines())
        elif _format == "tensorboard":
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            acc = EventAccumulator(str(tmp_path))
            acc.Reload()

            tb_values_logged = []
            for reservoir in [acc.scalars, acc.tensors, acc.images, acc.histograms, acc.compressed_histograms]:
                for k in reservoir.Keys():
                    tb_values_logged.append(f"{k}: {str(reservoir.Items(k))}")

            content = LogContent(_format, tb_values_logged)
            return content

    return read_fn


def test_set_logger(tmp_path):
    # set up logger
    new_logger = configure(str(tmp_path), ["stdout", "csv", "tensorboard"])
    # Default outputs with verbose=0
    model = A2C("MlpPolicy", "CartPole-v1", verbose=0).learn(4)
    assert model.logger.output_formats == []

    model = A2C("MlpPolicy", "CartPole-v1", verbose=0, tensorboard_log=str(tmp_path)).learn(4)
    assert str(tmp_path) in model.logger.dir
    assert isinstance(model.logger.output_formats[0], TensorBoardOutputFormat)

    # Check that env variable work
    new_tmp_path = str(tmp_path / "new_tmp")
    os.environ["SB3_LOGDIR"] = new_tmp_path
    model = A2C("MlpPolicy", "CartPole-v1", verbose=0).learn(4)
    assert model.logger.dir == new_tmp_path

    # Default outputs with verbose=1
    model = A2C("MlpPolicy", "CartPole-v1", verbose=1).learn(4)
    assert isinstance(model.logger.output_formats[0], HumanOutputFormat)
    # with tensorboard
    model = A2C("MlpPolicy", "CartPole-v1", verbose=1, tensorboard_log=str(tmp_path)).learn(4)
    assert isinstance(model.logger.output_formats[0], HumanOutputFormat)
    assert isinstance(model.logger.output_formats[1], TensorBoardOutputFormat)
    assert len(model.logger.output_formats) == 2
    model.learn(32)
    # set new logger
    model.set_logger(new_logger)
    # Check that the new logger is correctly setup
    assert isinstance(model.logger.output_formats[0], HumanOutputFormat)
    assert isinstance(model.logger.output_formats[1], CSVOutputFormat)
    assert isinstance(model.logger.output_formats[2], TensorBoardOutputFormat)
    assert len(model.logger.output_formats) == 3
    model.learn(32)

    model = A2C("MlpPolicy", "CartPole-v1", verbose=1)
    model.set_logger(new_logger)
    model.learn(32)
    # Check that the new logger is not overwritten
    assert isinstance(model.logger.output_formats[0], HumanOutputFormat)
    assert isinstance(model.logger.output_formats[1], CSVOutputFormat)
    assert isinstance(model.logger.output_formats[2], TensorBoardOutputFormat)
    assert len(model.logger.output_formats) == 3


def test_main(tmp_path):
    """
    tests for the logger module
    """
    logger = configure(None, ["stdout"])
    logger.info("hi")
    logger.debug("shouldn't appear")
    assert logger.level == INFO
    logger.set_level(DEBUG)
    assert logger.level == DEBUG
    logger.debug("should appear")
    logger = configure(folder=str(tmp_path))
    assert logger.dir == str(tmp_path)
    logger.record("a", 3)
    logger.record("b", 2.5)
    logger.dump()
    logger.record("b", -2.5)
    logger.record("a", 5.5)
    logger.dump()
    logger.info("^^^ should see a = 5.5")
    logger.record("f", "this text \n \r should appear in one line")
    logger.dump()
    logger.info('^^^ should see f = "this text \n \r should appear in one line"')
    logger.record_mean("b", -22.5)
    logger.record_mean("b", -44.4)
    logger.record("a", 5.5)
    logger.dump()

    logger.record("a", "longasslongasslongasslongasslongasslongassvalue")
    logger.dump()
    logger.warn("hey")
    logger.error("oh")


@pytest.mark.parametrize("_format", ["stdout", "log", "json", "csv", "tensorboard"])
def test_make_output(tmp_path, read_log, _format):
    """
    test make output

    :param _format: (str) output format
    """
    if _format == "tensorboard":
        # Skip if no tensorboard installed
        pytest.importorskip("tensorboard")

    writer = make_output_format(_format, tmp_path)
    writer.write(KEY_VALUES, KEY_EXCLUDED)
    assert not read_log(_format).empty
    writer.close()


def test_make_output_fail(tmp_path):
    """
    test value error on logger
    """
    with pytest.raises(ValueError):
        make_output_format("dummy_format", tmp_path)


@pytest.mark.parametrize("_format", ["stdout", "log", "json", "csv", "tensorboard"])
@pytest.mark.filterwarnings("ignore:Tried to write empty key-value dict")
def test_exclude_keys(tmp_path, read_log, _format):
    if _format == "tensorboard":
        # Skip if no tensorboard installed
        pytest.importorskip("tensorboard")

    writer = make_output_format(_format, tmp_path)
    writer.write(dict(some_tag=42), key_excluded=dict(some_tag=(_format)))
    writer.close()
    assert read_log(_format).empty


def test_report_video_to_tensorboard(tmp_path, read_log, capsys):
    pytest.importorskip("tensorboard")

    video = Video(frames=th.rand(1, 20, 3, 16, 16), fps=20)
    writer = make_output_format("tensorboard", tmp_path)
    writer.write({"video": video}, key_excluded={"video": ()})

    if is_moviepy_installed():
        assert not read_log("tensorboard").empty
    else:
        assert "moviepy" in capsys.readouterr().out
    writer.close()


def is_moviepy_installed():
    try:
        import moviepy  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


@pytest.mark.parametrize("unsupported_format", ["stdout", "log", "json", "csv"])
def test_report_video_to_unsupported_format_raises_error(tmp_path, unsupported_format):
    writer = make_output_format(unsupported_format, tmp_path)

    with pytest.raises(FormatUnsupportedError) as exec_info:
        video = Video(frames=th.rand(1, 20, 3, 16, 16), fps=20)
        writer.write({"video": video}, key_excluded={"video": ()})
    assert unsupported_format in str(exec_info.value)
    writer.close()


def test_report_image_to_tensorboard(tmp_path, read_log):
    pytest.importorskip("tensorboard")

    image = Image(image=th.rand(16, 16, 3), dataformats="HWC")
    writer = make_output_format("tensorboard", tmp_path)
    writer.write({"image": image}, key_excluded={"image": ()})

    assert not read_log("tensorboard").empty
    writer.close()


@pytest.mark.parametrize("unsupported_format", ["stdout", "log", "json", "csv"])
def test_report_image_to_unsupported_format_raises_error(tmp_path, unsupported_format):
    writer = make_output_format(unsupported_format, tmp_path)

    with pytest.raises(FormatUnsupportedError) as exec_info:
        image = Image(image=th.rand(16, 16, 3), dataformats="HWC")
        writer.write({"image": image}, key_excluded={"image": ()})
    assert unsupported_format in str(exec_info.value)
    writer.close()


def test_report_figure_to_tensorboard(tmp_path, read_log):
    pytest.importorskip("tensorboard")

    fig = plt.figure()
    fig.add_subplot().plot(np.random.random(3))
    figure = Figure(figure=fig, close=True)
    writer = make_output_format("tensorboard", tmp_path)
    writer.write({"figure": figure}, key_excluded={"figure": ()})

    assert not read_log("tensorboard").empty
    writer.close()


@pytest.mark.parametrize("unsupported_format", ["stdout", "log", "json", "csv"])
def test_report_figure_to_unsupported_format_raises_error(tmp_path, unsupported_format):
    writer = make_output_format(unsupported_format, tmp_path)

    with pytest.raises(FormatUnsupportedError) as exec_info:
        fig = plt.figure()
        fig.add_subplot().plot(np.random.random(3))
        figure = Figure(figure=fig, close=True)
        writer.write({"figure": figure}, key_excluded={"figure": ()})
    assert unsupported_format in str(exec_info.value)
    writer.close()


@pytest.mark.parametrize("unsupported_format", ["stdout", "log", "json", "csv"])
def test_report_hparam_to_unsupported_format_raises_error(tmp_path, unsupported_format):
    writer = make_output_format(unsupported_format, tmp_path)

    with pytest.raises(FormatUnsupportedError) as exec_info:
        hparam_dict = {"learning rate": np.random.random()}
        metric_dict = {"train/value_loss": 0}
        hparam = HParam(hparam_dict=hparam_dict, metric_dict=metric_dict)
        writer.write({"hparam": hparam}, key_excluded={"hparam": ()})
    assert unsupported_format in str(exec_info.value)
    writer.close()


def test_key_length(tmp_path):
    writer = make_output_format("stdout", tmp_path)
    assert writer.max_length == 36
    long_prefix = "a" * writer.max_length

    ok_dict = {
        # keys truncated but not aliased -- OK
        "a" + long_prefix: 42,
        "b" + long_prefix: 42,
        # values truncated and aliased -- also OK
        "foobar": long_prefix + "a",
        "fizzbuzz": long_prefix + "b",
    }
    ok_excluded = {k: None for k in ok_dict}
    writer.write(ok_dict, ok_excluded)

    long_key_dict = {
        long_prefix + "a": 42,
        "foobar": "sdf",
        long_prefix + "b": 42,
    }
    long_key_excluded = {k: None for k in long_key_dict}
    # keys truncated and aliased -- not OK
    with pytest.raises(ValueError, match="Key.*truncated"):
        writer.write(long_key_dict, long_key_excluded)

    # Just long enough to not be truncated now
    writer.max_length += 1
    writer.write(long_key_dict, long_key_excluded)


class TimeDelayEnv(gym.Env):
    """
    Gym env for testing FPS logging.
    """

    def __init__(self, delay: float = 0.01):
        super().__init__()
        self.delay = delay
        self.observation_space = gym.spaces.Box(low=-20.0, high=20.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        time.sleep(self.delay)
        obs = self.observation_space.sample()
        return obs, 0.0, True, {}


class InMemoryLogger(Logger):
    """
    Logger that keeps key/value pairs in memory without any writers.
    """

    def __init__(self):
        super().__init__("", [])

    def dump(self, step: int = 0) -> None:
        pass


@pytest.mark.parametrize("algo", [A2C, DQN])
def test_fps_logger(tmp_path, algo):
    logger = InMemoryLogger()
    max_fps = 1000
    env = TimeDelayEnv(1 / max_fps)
    model = algo("MlpPolicy", env, verbose=1)
    model.set_logger(logger)

    # fps should be at most max_fps
    model.learn(100, log_interval=1)
    assert max_fps / 10 <= logger.name_to_value["time/fps"] <= max_fps

    # second time, FPS should be the same
    model.learn(100, log_interval=1)
    assert max_fps / 10 <= logger.name_to_value["time/fps"] <= max_fps

    # Artificially increase num_timesteps to check
    # that fps computation is reset at each call to learn()
    model.num_timesteps = 20_000

    # third time, FPS should be the same
    model.learn(100, log_interval=1, reset_num_timesteps=False)
    assert max_fps / 10 <= logger.name_to_value["time/fps"] <= max_fps


@pytest.mark.parametrize("algo", [A2C, DQN])
def test_fps_no_div_zero(algo):
    """Set time to constant and train algorithm to check no division by zero error.

    Time can appear to be constant during short runs on platforms with low-precision
    timers. We should avoid division by zero errors e.g. when computing FPS in
    this situation."""
    with mock.patch("time.time", lambda: 42.0):
        with mock.patch("time.time_ns", lambda: 42.0):
            model = algo("MlpPolicy", "CartPole-v1")
            model.learn(total_timesteps=100)


def test_human_output_format_no_crash_on_same_keys_different_tags():
    o = HumanOutputFormat(sys.stdout, max_length=60)
    o.write(
        {"key1/foo": "value1", "key1/bar": "value2", "key2/bizz": "value3", "key2/foo": "value4"},
        {"key1/foo": None, "key2/bizz": None, "key1/bar": None, "key2/foo": None},
    )
