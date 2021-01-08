from typing import Sequence

import numpy as np
import pytest
import torch as th
from matplotlib import pyplot as plt
from pandas.errors import EmptyDataError

from stable_baselines3.common.logger import (
    DEBUG,
    Figure,
    FormatUnsupportedError,
    Image,
    ScopedConfigure,
    Video,
    configure,
    debug,
    dump,
    error,
    info,
    make_output_format,
    read_csv,
    read_json,
    record,
    record_dict,
    record_mean,
    reset,
    set_level,
    warn,
)

KEY_VALUES = {
    "test": 1,
    "b": -3.14,
    "8": 9.9,
    "l": [1, 2],
    "a": np.array([1, 2, 3]),
    "f": np.array(1),
    "g": np.array([[[1]]]),
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


def test_main(tmp_path):
    """
    tests for the logger module
    """
    info("hi")
    debug("shouldn't appear")
    set_level(DEBUG)
    debug("should appear")
    configure(folder=str(tmp_path))
    record("a", 3)
    record("b", 2.5)
    dump()
    record("b", -2.5)
    record("a", 5.5)
    dump()
    info("^^^ should see a = 5.5")
    record_mean("b", -22.5)
    record_mean("b", -44.4)
    record("a", 5.5)
    dump()
    with ScopedConfigure(None, None):
        info("^^^ should see b = 33.3")

    with ScopedConfigure(str(tmp_path / "test-logger"), ["json"]):
        record("b", -2.5)
        dump()

    reset()
    record("a", "longasslongasslongasslongasslongasslongassvalue")
    dump()
    warn("hey")
    error("oh")
    record_dict({"test": 1})


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
