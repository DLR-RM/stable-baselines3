import os
import os.path
from typing import Callable

import numpy as np
from gymnasium import error, logger

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


class VecVideoRecorder(VecEnvWrapper):
    """
    Wraps a VecEnv or VecEnvWrapper object to record rendered image as mp4 video.
    It requires ffmpeg or avconv to be installed on the machine.

    Note: for now it only allows to record one video and all videos
    must have at least two frames.

    The video recorder code was adapted from Gymnasium v1.0.

    :param venv:
    :param video_folder: Where to save videos
    :param record_video_trigger: Function that defines when to start recording.
                                        The function takes the current number of step,
                                        and returns whether we should start recording or not.
    :param video_length:  Length of recorded videos
    :param name_prefix: Prefix to the video name
    """

    video_name: str
    video_path: str

    def __init__(
        self,
        venv: VecEnv,
        video_folder: str,
        record_video_trigger: Callable[[int], bool],
        video_length: int = 200,
        name_prefix: str = "rl-video",
    ):
        VecEnvWrapper.__init__(self, venv)

        self.env = venv
        # Temp variable to retrieve metadata
        temp_env = venv

        # Unwrap to retrieve metadata dict
        # that will be used by gym recorder
        while isinstance(temp_env, VecEnvWrapper):
            temp_env = temp_env.venv

        if isinstance(temp_env, DummyVecEnv) or isinstance(temp_env, SubprocVecEnv):
            metadata = temp_env.get_attr("metadata")[0]
        else:  # pragma: no cover # assume gym interface
            metadata = temp_env.metadata

        self.env.metadata = metadata
        assert self.env.render_mode == "rgb_array", f"The render_mode must be 'rgb_array', not {self.env.render_mode}"

        self.frames_per_sec = self.env.metadata.get("render_fps", 30)

        self.record_video_trigger = record_video_trigger
        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames: list[np.ndarray] = []

        try:
            import moviepy  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise error.DependencyNotInstalled("MoviePy is not installed, run `pip install 'gymnasium[other]'`") from e

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        if self._video_enabled():
            self._start_video_recorder()
        return obs

    def _start_video_recorder(self) -> None:
        # Update video name and path
        self.video_name = f"{self.name_prefix}-step-{self.step_id}-to-step-{self.step_id + self.video_length}.mp4"
        self.video_path = os.path.join(self.video_folder, self.video_name)
        self._start_recording()
        self._capture_frame()

    def _video_enabled(self) -> bool:
        return self.record_video_trigger(self.step_id)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()

        self.step_id += 1
        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) > self.video_length:
                print(f"Saving video to {self.video_path}")
                self._stop_recording()
        elif self._video_enabled():
            self._start_video_recorder()

        return obs, rewards, dones, infos

    def _capture_frame(self) -> None:
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        frame = self.env.render()

        if isinstance(frame, np.ndarray):
            self.recorded_frames.append(frame)
        else:
            self._stop_recording()
            logger.warn(
                f"Recording stopped: expected type of frame returned by render to be a numpy array, got instead {type(frame)}."
            )

    def close(self) -> None:
        """Closes the wrapper then the video recorder."""
        VecEnvWrapper.close(self)
        if self.recording:  # pragma: no cover
            self._stop_recording()

    def _start_recording(self) -> None:
        """Start a new recording. If it is already recording, stops the current recording before starting the new one."""
        if self.recording:  # pragma: no cover
            self._stop_recording()

        self.recording = True

    def _stop_recording(self) -> None:
        """Stop current recording and saves the video."""
        assert self.recording, "_stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:  # pragma: no cover
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            clip.write_videofile(self.video_path)

        self.recorded_frames = []
        self.recording = False

    def __del__(self) -> None:
        """Warn the user in case last video wasn't saved."""
        if len(self.recorded_frames) > 0:  # pragma: no cover
            logger.warn("Unable to save last video! Did you call close()?")
