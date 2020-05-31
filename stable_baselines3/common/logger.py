import sys
import datetime
import json
import os
import tempfile
import warnings
from collections import defaultdict
from typing import Dict, List, TextIO, Union, Any, Optional, Tuple

import pandas
import numpy as np
import torch as th
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50


class KVWriter(object):
    """
    Key Value writer
    """

    def write(self, key_values: Dict[str, Any],
              key_excluded: Dict[str, Union[str, Tuple[str, ...]]], step: int = 0) -> None:
        """
        Write a dictionary to file

        :param key_values: (dict)
        :param key_excluded: (dict)
        :param step: (int)
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Close owned resources
        """
        raise NotImplementedError


class SeqWriter(object):
    """
    sequence writer
    """

    def write_sequence(self, sequence: List):
        """
        write_sequence an array to file

        :param sequence: (list)
        """
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file: Union[str, TextIO]):
        """
        log to a file, in a human readable format

        :param filename_or_file: (str or File) the file to write the log to
        """
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, 'wt')
            self.own_file = True
        else:
            assert hasattr(filename_or_file, 'write'), f'Expected file or str, got {filename_or_file}'
            self.file = filename_or_file
            self.own_file = False

    def write(self, key_values: Dict, key_excluded: Dict, step: int = 0) -> None:
        # Create strings for printing
        key2str = {}
        tag = None
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):

            if excluded is not None and 'stdout' in excluded:
                continue

            if isinstance(value, float):
                # Align left
                value_str = f'{value:<8.3g}'
            else:
                value_str = str(value)

            if key.find('/') > 0:  # Find tag and add it to the dict
                tag = key[:key.find('/') + 1]
                key2str[self._truncate(tag)] = ''
            # Remove tag from key
            if tag is not None and tag in key:
                key = str('   ' + key[len(tag):])

            key2str[self._truncate(key)] = self._truncate(value_str)

        # Find max widths
        if len(key2str) == 0:
            warnings.warn('Tried to write empty key-value dict')
            return
        else:
            key_width = max(map(len, key2str.keys()))
            val_width = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (key_width + val_width + 7)
        lines = [dashes]
        for key, value in key2str.items():
            key_space = ' ' * (key_width - len(key))
            val_space = ' ' * (val_width - len(value))
            lines.append(f"| {key}{key_space} | {value}{val_space} |")
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()

    @classmethod
    def _truncate(cls, string: str, max_length: int = 23) -> str:
        return string[:max_length - 3] + '...' if len(string) > max_length else string

    def write_sequence(self, sequence: List) -> None:
        sequence = list(sequence)
        for i, elem in enumerate(sequence):
            self.file.write(elem)
            if i < len(sequence) - 1:  # add space unless this is the last one
                self.file.write(' ')
        self.file.write('\n')
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename: str):
        """
        log to a file, in the JSON format

        :param filename: (str) the file to write the log to
        """
        self.file = open(filename, 'wt')

    def write(self, key_values: Dict[str, Any],
              key_excluded: Dict[str, Union[str, Tuple[str, ...]]], step: int = 0) -> None:
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):

            if excluded is not None and 'json' in excluded:
                continue

            if hasattr(value, 'dtype'):
                if value.shape == () or len(value) == 1:
                    # if value is a dimensionless numpy array or of length 1, serialize as a float
                    key_values[key] = float(value)
                else:
                    # otherwise, a value is a numpy array, serialize as a list or nested lists
                    key_values[key] = value.tolist()
        self.file.write(json.dumps(key_values) + '\n')
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """

        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename: str):
        """
        log to a file, in a CSV format

        :param filename: (str) the file to write the log to
        """

        self.file = open(filename, 'w+t')
        self.keys = []
        self.separator = ','

    def write(self, key_values: Dict[str, Any],
              key_excluded: Dict[str, Union[str, Tuple[str, ...]]], step: int = 0) -> None:
        # Add our current row to the history
        extra_keys = key_values.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, key) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(key)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.separator * len(extra_keys))
                self.file.write('\n')
        for i, key in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            value = key_values.get(key)
            if value is not None:
                self.file.write(str(value))
        self.file.write('\n')
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    def __init__(self, folder: str):
        """
        Dumps key/value pairs into TensorBoard's numeric format.

        :param folder: (str) the folder to write the log to
        """
        assert SummaryWriter is not None, ("tensorboard is not installed, you can use "
                                           "pip install tensorboard to do so")
        self.writer = SummaryWriter(log_dir=folder)

    def write(self, key_values: Dict[str, Any],
              key_excluded: Dict[str, Union[str, Tuple[str, ...]]], step: int = 0) -> None:

        for (key, value), (_, excluded) in zip(sorted(key_values.items()),
                                               sorted(key_excluded.items())):

            if excluded is not None and 'tensorboard' in excluded:
                continue

            if isinstance(value, np.ScalarType):
                self.writer.add_scalar(key, value, step)

            if isinstance(value, th.Tensor):
                self.writer.add_histogram(key, value, step)

        # Flush the output to the file
        self.writer.flush()

    def close(self) -> None:
        """
        closes the file
        """
        if self.writer:
            self.writer.close()
            self.writer = None


def make_output_format(_format: str, log_dir: str, log_suffix: str = '') -> KVWriter:
    """
    return a logger for the requested format

    :param _format: (str) the requested format to log to ('stdout', 'log', 'json' or 'csv' or 'tensorboard')
    :param log_dir: (str) the logging directory
    :param log_suffix: (str) the suffix for the log file
    :return: (KVWriter) the logger
    """
    os.makedirs(log_dir, exist_ok=True)
    if _format == 'stdout':
        return HumanOutputFormat(sys.stdout)
    elif _format == 'log':
        return HumanOutputFormat(os.path.join(log_dir, f'log{log_suffix}.txt'))
    elif _format == 'json':
        return JSONOutputFormat(os.path.join(log_dir, f'progress{log_suffix}.json'))
    elif _format == 'csv':
        return CSVOutputFormat(os.path.join(log_dir, f'progress{log_suffix}.csv'))
    elif _format == 'tensorboard':
        return TensorBoardOutputFormat(log_dir)
    else:
        raise ValueError(f'Unknown format specified: {_format}')


# ================================================================
# API
# ================================================================

def record(key: str, value: Any,
           exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.

    :param key: (Any) save to log this key
    :param value: (Any) save to log this value
    :param exclude: (str or tuple) outputs to be excluded
    """
    Logger.CURRENT.record(key, value, exclude)


def record_mean(key: str, value: Union[int, float],
                exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
    """
    The same as record(), but if called many times, values averaged.

    :param key: (Any) save to log this key
    :param value: (Number) save to log this value
    :param exclude: (str or tuple) outputs to be excluded
    """
    Logger.CURRENT.record_mean(key, value, exclude)


def record_dict(key_values: Dict[str, Any]) -> None:
    """
    Log a dictionary of key-value pairs.

    :param key_values: (dict) the list of keys and values to save to log
    """
    for key, value in key_values.items():
        record(key, value)


def dump(step: int = 0) -> None:
    """
    Write all of the diagnostics from the current iteration
    """
    Logger.CURRENT.dump(step)


def get_log_dict() -> Dict:
    """
    get the key values logs

    :return: (dict) the logged values
    """
    return Logger.CURRENT.name_to_value


def log(*args, level: int = INFO) -> None:
    """
    Write the sequence of args, with no separators,
    to the console and output files (if you've configured an output file).

    level: int. (see logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.

    :param args: (list) log the arguments
    :param level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
    """
    Logger.CURRENT.log(*args, level=level)


def debug(*args) -> None:
    """
    Write the sequence of args, with no separators,
    to the console and output files (if you've configured an output file).
    Using the DEBUG level.

    :param args: (list) log the arguments
    """
    log(*args, level=DEBUG)


def info(*args) -> None:
    """
    Write the sequence of args, with no separators,
    to the console and output files (if you've configured an output file).
    Using the INFO level.

    :param args: (list) log the arguments
    """
    log(*args, level=INFO)


def warn(*args) -> None:
    """
    Write the sequence of args, with no separators,
    to the console and output files (if you've configured an output file).
    Using the WARN level.

    :param args: (list) log the arguments
    """
    log(*args, level=WARN)


def error(*args) -> None:
    """
    Write the sequence of args, with no separators,
    to the console and output files (if you've configured an output file).
    Using the ERROR level.

    :param args: (list) log the arguments
    """
    log(*args, level=ERROR)


def set_level(level: int) -> None:
    """
    Set logging threshold on current logger.

    :param level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
    """
    Logger.CURRENT.set_level(level)


def get_level() -> int:
    """
    Get logging threshold on current logger.
    :return: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
    """
    return Logger.CURRENT.level


def get_dir() -> str:
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)

    :return: (str) the logging directory
    """
    return Logger.CURRENT.get_dir()


record_tabular = record
dump_tabular = dump


# ================================================================
# Backend
# ================================================================

class Logger(object):
    # A logger with no output files. (See right below class definition)
    #  So that you can still log to the terminal without setting up any output files
    DEFAULT = None
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, folder: Optional[str], output_formats: List[KVWriter]):
        """
        the logger class

        :param folder: (str) the logging location
        :param output_formats: ([str]) the list of output format
        """
        self.name_to_value = defaultdict(float)  # values this iteration
        self.name_to_count = defaultdict(int)
        self.name_to_excluded = defaultdict(str)
        self.level = INFO
        self.dir = folder
        self.output_formats = output_formats

    # Logging API, forwarded
    # ----------------------------------------
    def record(self, key: str, value: Any,
               exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: (Any) save to log this key
        :param value: (Any) save to log this value
        :param exclude: (str or tuple) outputs to be excluded
        """
        self.name_to_value[key] = value
        self.name_to_excluded[key] = exclude

    def record_mean(self, key: str, value: Any,
                    exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        """
        The same as record(), but if called many times, values averaged.

        :param key: (Any) save to log this key
        :param value: (Number) save to log this value
        :param exclude: (str or tuple) outputs to be excluded
        """
        if value is None:
            self.name_to_value[key] = None
            return
        old_val, count = self.name_to_value[key], self.name_to_count[key]
        self.name_to_value[key] = old_val * count / (count + 1) + value / (count + 1)
        self.name_to_count[key] = count + 1
        self.name_to_excluded[key] = exclude

    def dump(self, step: int = 0) -> None:
        """
        Write all of the diagnostics from the current iteration
        """
        if self.level == DISABLED:
            return
        for _format in self.output_formats:
            if isinstance(_format, KVWriter):
                _format.write(self.name_to_value, self.name_to_excluded, step)

        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()

    def log(self, *args, level: int = INFO) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).

        level: int. (see logger.py docs) If the global logger level is higher than
                    the level argument here, don't print to stdout.

        :param args: (list) log the arguments
        :param level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
        """
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level: int) -> None:
        """
        Set logging threshold on current logger.

        :param level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
        """
        self.level = level

    def get_dir(self) -> str:
        """
        Get directory that log files are being written to.
        will be None if there is no output directory (i.e., if you didn't call start)

        :return: (str) the logging directory
        """
        return self.dir

    def close(self) -> None:
        """
        closes the file
        """
        for _format in self.output_formats:
            _format.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args) -> None:
        """
        log to the requested format outputs

        :param args: (list) the arguments to log
        """
        for _format in self.output_formats:
            if isinstance(_format, SeqWriter):
                _format.write_sequence(map(str, args))


# Initialize logger
Logger.DEFAULT = Logger.CURRENT = Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout)])


def configure(folder: Optional[str] = None, format_strings: Optional[List[str]] = None) -> None:
    """
    configure the current logger

    :param folder: (Optional[str]) the save location
        (if None, $SB3_LOGDIR, if still None, tempdir/baselines-[date & time])
    :param format_strings: (Optional[List[str]]) the output logging format
        (if None, $SB3_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
    """
    if folder is None:
        folder = os.getenv('SB3_LOGDIR')
    if folder is None:
        folder = os.path.join(tempfile.gettempdir(), datetime.datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)

    log_suffix = ''
    if format_strings is None:
        format_strings = os.getenv('SB3_LOG_FORMAT', 'stdout,log,csv').split(',')

    format_strings = filter(None, format_strings)
    output_formats = [make_output_format(f, folder, log_suffix) for f in format_strings]

    Logger.CURRENT = Logger(folder=folder, output_formats=output_formats)
    log(f'Logging to {folder}')


def reset() -> None:
    """
    reset the current logger
    """
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log('Reset logger')


class ScopedConfigure(object):
    def __init__(self, folder: Optional[str] = None, format_strings: Optional[List[str]] = None):
        """
        Class for using context manager while logging

        usage:
        with ScopedConfigure(folder=None, format_strings=None):
            {code}

        :param folder: (str) the logging folder
        :param format_strings: ([str]) the list of output logging format
        """
        self.dir = folder
        self.format_strings = format_strings
        self.prev_logger = None

    def __enter__(self) -> None:
        self.prev_logger = Logger.CURRENT
        configure(folder=self.dir, format_strings=self.format_strings)

    def __exit__(self, *args) -> None:
        Logger.CURRENT.close()
        Logger.CURRENT = self.prev_logger


# ================================================================
# Readers
# ================================================================

def read_json(filename: str) -> pandas.DataFrame:
    """
    read a json file using pandas

    :param filename: (str) the file path to read
    :return: (pandas.DataFrame) the data in the json
    """
    data = []
    with open(filename, 'rt') as file_handler:
        for line in file_handler:
            data.append(json.loads(line))
    return pandas.DataFrame(data)


def read_csv(filename: str) -> pandas.DataFrame:
    """
    read a csv file using pandas

    :param filename: (str) the file path to read
    :return: (pandas.DataFrame) the data in the csv
    """
    return pandas.read_csv(filename, index_col=None, comment='#')
