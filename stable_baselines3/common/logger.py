import sys
import datetime
import json
import os
import tempfile
import warnings
from collections import defaultdict
from typing import Dict, List, TextIO, Union, Any, Optional

import pandas

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50


class KVWriter(object):
    """
    Key Value writer
    """

    def writekvs(self, kvs: Dict) -> None:
        """
        write a dictionary to file

        :param kvs: (dict)
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

    def writeseq(self, seq: List):
        """
        write an array to file

        :param seq: (list)
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

    def writekvs(self, kvs: Dict) -> None:
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, float):
                # Align left
                val_str = f'{val:<8.3g}'
            else:
                val_str = str(val)
            key2str[self._truncate(key)] = self._truncate(val_str)

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
        for (key, val) in sorted(key2str.items()):
            key_space = ' ' * (key_width - len(key))
            val_space = ' ' * (val_width - len(val))
            lines.append(f"| {key}{key_space} | {val}{val_space} |")
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()

    @classmethod
    def _truncate(cls, string: str) -> str:
        return string[:20] + '...' if len(string) > 23 else string

    def writeseq(self, seq: List) -> None:
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
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

    def writekvs(self, kvs: Dict) -> None:
        for key, value in sorted(kvs.items()):
            if hasattr(value, 'dtype'):
                if value.shape == () or len(value) == 1:
                    # if value is a dimensionless numpy array or of length 1, serialize as a float
                    kvs[key] = float(value)
                else:
                    # otherwise, a value is a numpy array, serialize as a list or nested lists
                    kvs[key] = value.tolist()
        self.file.write(json.dumps(kvs) + '\n')
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
        self.sep = ','

    def writekvs(self, kvs: Dict) -> None:
        # Add our current row to the history
        extra_keys = kvs.keys() - self.keys
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
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
        for i, key in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            value = kvs.get(key)
            if value is not None:
                self.file.write(str(value))
        self.file.write('\n')
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        self.file.close()


def valid_float_value(value: Any) -> bool:
    """
    Returns True if the value can be successfully cast into a float

    :param value: (Any) the value to check
    :return: (bool)
    """
    try:
        float(value)
        return True
    except TypeError:
        return False


def make_output_format(_format: str, log_dir: str, log_suffix: str = '') -> KVWriter:
    """
    return a logger for the requested format

    :param _format: (str) the requested format to log to ('stdout', 'log', 'json' or 'csv')
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
    else:
        raise ValueError(f'Unknown format specified: {_format}')


# ================================================================
# API
# ================================================================

def logkv(key: Any, val: Any) -> None:
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.

    :param key: (Any) save to log this key
    :param val: (Any) save to log this value
    """
    Logger.CURRENT.logkv(key, val)


def logkv_mean(key: Any, val: Union[int, float]) -> None:
    """
    The same as logkv(), but if called many times, values averaged.

    :param key: (Any) save to log this key
    :param val: (Number) save to log this value
    """
    Logger.CURRENT.logkv_mean(key, val)


def logkvs(key_values: Dict) -> None:
    """
    Log a dictionary of key-value pairs

    :param key_values: (dict) the list of keys and values to save to log
    """
    for key, value in key_values.items():
        logkv(key, value)


def dumpkvs() -> None:
    """
    Write all of the diagnostics from the current iteration
    """
    Logger.CURRENT.dumpkvs()


def getkvs() -> Dict:
    """
    get the key values logs

    :return: (dict) the logged values
    """
    return Logger.CURRENT.name2val


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


record_tabular = logkv
dump_tabular = dumpkvs


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
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = folder
        self.output_formats = output_formats

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key: Any, val: Any) -> None:
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: (Any) save to log this key
        :param val: (Any) save to log this value
        """
        self.name2val[key] = val

    def logkv_mean(self, key: Any, val: Any) -> None:
        """
        The same as logkv(), but if called many times, values averaged.

        :param key: (Any) save to log this key
        :param val: (Number) save to log this value
        """
        if val is None:
            self.name2val[key] = None
            return
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self) -> None:
        """
        Write all of the diagnostics from the current iteration
        """
        if self.level == DISABLED:
            return
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(self.name2val)
        self.name2val.clear()
        self.name2cnt.clear()

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
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args) -> None:
        """
        log to the requested format outputs

        :param args: (list) the arguments to log
        """
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))


# Initialize logger
Logger.DEFAULT = Logger.CURRENT = Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout)])


def configure(folder: Optional[str] = None, format_strs: Optional[List[str]] = None) -> None:
    """
    configure the current logger

    :param folder: (Optional[str]) the save location
        (if None, $BASELINES_LOGDIR, if still None, tempdir/baselines-[date & time])
    :param format_strs: (Optional[List[str]]) the output logging format
        (if None, $BASELINES_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
    """
    if folder is None:
        folder = os.getenv('BASELINES_LOGDIR')
    if folder is None:
        folder = os.path.join(tempfile.gettempdir(), datetime.datetime.now().strftime("baselines-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)

    log_suffix = ''
    if format_strs is None:
        format_strs = os.getenv('BASELINES_LOG_FORMAT', 'stdout,log,csv').split(',')

    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, folder, log_suffix) for f in format_strs]

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
    def __init__(self, folder: Optional[str] = None, format_strs: Optional[List[str]] = None):
        """
        Class for using context manager while logging

        usage:
        with ScopedConfigure(folder=None, format_strs=None):
            {code}

        :param folder: (str) the logging folder
        :param format_strs: ([str]) the list of output logging format
        """
        self.dir = folder
        self.format_strs = format_strs
        self.prevlogger = None

    def __enter__(self) -> None:
        self.prevlogger = Logger.CURRENT
        configure(folder=self.dir, format_strs=self.format_strs)

    def __exit__(self, *args) -> None:
        Logger.CURRENT.close()
        Logger.CURRENT = self.prevlogger


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
