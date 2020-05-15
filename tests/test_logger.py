import os
import shutil

import pytest
import numpy as np

from stable_baselines3.common.logger import (make_output_format, read_csv, read_json, DEBUG, ScopedConfigure,
                                             info, debug, set_level, configure, logkv, logkvs,
                                             dumpkvs, logkv_mean, warn, error, reset)

KEY_VALUES = {
    "test": 1,
    "b": -3.14,
    "8": 9.9,
    "l": [1, 2],
    "a": np.array([1, 2, 3]),
    "f": np.array(1),
    "g": np.array([[[1]]]),
}

LOG_DIR = '/tmp/stable_baselines3/'


def test_main():
    """
    tests for the logger module
    """
    info("hi")
    debug("shouldn't appear")
    set_level(DEBUG)
    debug("should appear")
    folder = "/tmp/testlogging"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    configure(folder=folder)
    logkv("a", 3)
    logkv("b", 2.5)
    dumpkvs()
    logkv("b", -2.5)
    logkv("a", 5.5)
    dumpkvs()
    info("^^^ should see a = 5.5")
    logkv_mean("b", -22.5)
    logkv_mean("b", -44.4)
    logkv("a", 5.5)
    dumpkvs()
    with ScopedConfigure(None, None):
        info("^^^ should see b = 33.3")

    with ScopedConfigure("/tmp/test-logger/", ["json"]):
        logkv("b", -2.5)
        dumpkvs()

    reset()
    logkv("a", "longasslongasslongasslongasslongasslongassvalue")
    dumpkvs()
    warn("hey")
    error("oh")
    logkvs({"test": 1})


@pytest.mark.parametrize('_format', ['stdout', 'log', 'json', 'csv'])
def test_make_output(_format):
    """
    test make output

    :param _format: (str) output format
    """
    writer = make_output_format(_format, LOG_DIR)
    writer.writekvs(KEY_VALUES)
    if _format == "csv":
        read_csv(LOG_DIR + 'progress.csv')
    elif _format == 'json':
        read_json(LOG_DIR + 'progress.json')
    writer.close()


def test_make_output_fail():
    """
    test value error on logger
    """
    with pytest.raises(ValueError):
        make_output_format('dummy_format', LOG_DIR)
