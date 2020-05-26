import os
import shutil

import pytest
import numpy as np

from stable_baselines3.common.logger import (make_output_format, read_csv, read_json, DEBUG, ScopedConfigure,
                                             info, debug, set_level, configure, record, record_dict,
                                             dump, record_mean, warn, error, reset)

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

    with ScopedConfigure("/tmp/test-logger/", ["json"]):
        record("b", -2.5)
        dump()

    reset()
    record("a", "longasslongasslongasslongasslongasslongassvalue")
    dump()
    warn("hey")
    error("oh")
    record_dict({"test": 1})


@pytest.mark.parametrize('_format', ['stdout', 'log', 'json', 'csv'])
def test_make_output(_format):
    """
    test make output

    :param _format: (str) output format
    """
    writer = make_output_format(_format, LOG_DIR)
    writer.write(KEY_VALUES, KEY_EXCLUDED)
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
