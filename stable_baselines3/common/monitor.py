__all__ = ["Monitor", "ResultsWriter", "get_monitor_files", "load_results"]

import csv
import json
import os
import time
from glob import glob
from typing import Any, Optional, SupportsFloat, Union

import gymnasium as gym
import pandas
from gymnasium.core import ActType, ObsType


class Monitor(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    """

    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: tuple[str, ...] = (),
        info_keywords: tuple[str, ...] = (),
        override_existing: bool = True,
    ):
        super().__init__(env=env)
        self.t_start = time.time()
        self.results_writer = None
        if filename is not None:
            env_id = env.spec.id if env.spec is not None else None
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": str(env_id)},
                extra_keys=reset_keywords + info_keywords,
                override_existing=override_existing,
            )

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards: list[float] = []
        self.needs_reset = True
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_times: list[float] = []
        self.total_steps = 0
        # extra info about the current episode, that was passed in during reset()
        self.current_reset_info: dict[str, Any] = {}

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(f"Expected you to pass keyword argument {key} into reset")
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))
        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """
        Closes the environment
        """
        super().close()
        if self.results_writer is not None:
            self.results_writer.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> list[float]:
        """
        Returns the rewards of all the episodes

        :return:
        """
        return self.episode_returns

    def get_episode_lengths(self) -> list[int]:
        """
        Returns the number of timesteps of all the episodes

        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> list[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return:
        """
        return self.episode_times


class LoadMonitorResultsError(Exception):
    """
    Raised when loading the monitor log fails.
    """

    pass


class ResultsWriter:
    """
    A result writer that saves the data from the `Monitor` class

    :param filename: the location to save a log file. When it does not end in
        the string ``"monitor.csv"``, this suffix will be appended to it
    :param header: the header dictionary object of the saved csv
    :param extra_keys: the extra information to log, typically is composed of
        ``reset_keywords`` and ``info_keywords``
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    """

    def __init__(
        self,
        filename: str = "",
        header: Optional[dict[str, Union[float, str]]] = None,
        extra_keys: tuple[str, ...] = (),
        override_existing: bool = True,
    ):
        if header is None:
            header = {}
        if not filename.endswith(Monitor.EXT):
            if os.path.isdir(filename):
                filename = os.path.join(filename, Monitor.EXT)
            else:
                filename = filename + "." + Monitor.EXT
        filename = os.path.realpath(filename)
        # Create (if any) missing filename directories
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Append mode when not overriding existing file
        mode = "w" if override_existing else "a"
        # Prevent newline issue on Windows, see GH issue #692
        self.file_handler = open(filename, f"{mode}t", newline="\n")
        self.logger = csv.DictWriter(self.file_handler, fieldnames=("r", "l", "t", *extra_keys))
        if override_existing:
            self.file_handler.write(f"#{json.dumps(header)}\n")
            self.logger.writeheader()

        self.file_handler.flush()

    def write_row(self, epinfo: dict[str, float]) -> None:
        """
        Write row of monitor data to csv log file.

        :param epinfo: the information on episodic return, length, and time
        """
        if self.logger:
            self.logger.writerow(epinfo)
            self.file_handler.flush()

    def close(self) -> None:
        """
        Close the file handler
        """
        self.file_handler.close()


def get_monitor_files(path: str) -> list[str]:
    """
    get all the monitor files in the given path

    :param path: the logging folder
    :return: the log files
    """
    return glob(os.path.join(path, "*" + Monitor.EXT))


def load_results(path: str) -> pandas.DataFrame:
    """
    Load all Monitor logs from a given directory path matching ``*monitor.csv``

    :param path: the directory path containing the log file(s)
    :return: the logged data
    """
    monitor_files = get_monitor_files(path)
    if len(monitor_files) == 0:
        raise LoadMonitorResultsError(f"No monitor files of the form *{Monitor.EXT} found in {path}")
    data_frames, headers = [], []
    for file_name in monitor_files:
        with open(file_name) as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == "#"
            header = json.loads(first_line[1:])
            data_frame = pandas.read_csv(file_handler, index_col=None)
            headers.append(header)
            data_frame["t"] += header["t_start"]
        data_frames.append(data_frame)
    data_frame = pandas.concat(data_frames)
    data_frame.sort_values("t", inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame["t"] -= min(header["t_start"] for header in headers)
    return data_frame
