import multiprocessing
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def _worker(  # noqa: C901
    remote: multiprocessing.connection.Connection,
    parent_remote: multiprocessing.connection.Connection,
    env_fns_wrapper: CloudpickleWrapper,
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    envs = [_patch_env(env_fn()) for env_fn in env_fns_wrapper.var]
    reset_info: dict[str, Any] | None = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                results = []
                for env, action in zip(envs, data, strict=True):
                    observation, reward, terminated, truncated, info = env.step(action)
                    # convert to SB3 VecEnv api
                    done = terminated or truncated
                    info["TimeLimit.truncated"] = truncated and not terminated
                    if done:
                        # save final observation where user can get it, then reset
                        info["terminal_observation"] = observation
                        observation, reset_info = env.reset()
                    results.append((observation, reward, done, info, reset_info))
                remote.send(results)
            elif cmd == "reset":
                assert len(data) == len(envs)
                remote.send([
                    env.reset(**params)
                    for env, params in zip(envs, data, strict=True)
                ])  # fmt: skip
            elif cmd == "render":
                remote.send([env.render() for env in envs])
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((envs[0].observation_space, envs[0].action_space))
            elif cmd == "env_method":
                inds, method_name, method_args, method_kwargs = data
                if inds is None:
                    inds = list(range(len(envs)))
                remote.send([
                    envs[ind].get_wrapper_attr(method_name)(*method_args, **method_kwargs)
                    for ind in inds
                ])  # fmt: skip
            elif cmd == "get_attr":
                inds, attr_name = data
                if inds is None:
                    inds = list(range(len(envs)))
                remote.send([
                    envs[ind].get_wrapper_attr(attr_name)
                    for ind in inds
                ])  # fmt: skip
            elif cmd == "has_attr":
                attr_name = data
                if inds is None:
                    inds = list(range(len(envs)))

                def _has_att(env, att_name):
                    try:
                        env.get_wrapper_attr(att_name)
                        return True
                    except AttributeError:
                        return False

                remote.send([
                    _has_att(envs[ind], attr_name)
                    for ind in inds
                ])  # fmt: skip
            elif cmd == "set_attr":
                inds, attr_name, attr_value = data
                if inds is None:
                    inds = list(range(len(envs)))
                remote.send([
                    setattr(envs[ind], attr_name, attr_value)  # type: ignore[func-returns-value]
                    for ind in inds
                ])  # fmt: skip
            elif cmd == "is_wrapped":
                inds, wrapper_class = data
                if inds is None:
                    inds = list(range(len(envs)))
                remote.send([
                    is_wrapped(envs[ind], wrapper_class)
                    for ind in inds
                ])  # fmt: skip
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break
        except KeyboardInterrupt:
            break


class PoolVecEnv(VecEnv):
    """
    Creates a multiple-worker-based wrapper for multiple environments. Compared to `SubprocVecEnv`,
    each worker here can run multiple environments. This reduces the number of OS context switches
    and latency of data exchange between the main process and the workers. This is especially useful for
    simulating a large number of environments (i.e., high batch size) where each environment is
    relatively lightweight (e.g. CartPole, Atari).

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param num_worker: Number of worker processes to use. If =0, use the number of CPU cores available.
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        num_worker: int = 0,
        start_method: str | None = None,
    ):
        if num_worker == 0:
            num_worker = multiprocessing.cpu_count()

        self.waiting = False
        self.closed = False

        n_envs = len(env_fns)

        self.num_worker = min(n_envs, num_worker)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in multiprocessing.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = multiprocessing.get_context(start_method)

        env_fns_chunks = [[] for _ in range(self.num_worker)]  # type: ignore[var-annotated]
        self.env_inds_chunks = [[] for _ in range(self.num_worker)]  # type: ignore[var-annotated]
        for ind, env_fn in enumerate(env_fns):
            env_fns_chunks[ind % self.num_worker].append(env_fn)
            self.env_inds_chunks[ind % self.num_worker].append(ind)
        self.env_ind_to_worker_indpair = lambda ind: (ind % self.num_worker, ind // self.num_worker)
        self.worker_indpair_to_env_ind = lambda indpair: self.num_worker * indpair[1] + indpair[0]

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_worker)], strict=True)

        self.processes = []
        for work_remote, remote, env_fns_chunk in zip(self.work_remotes, self.remotes, env_fns_chunks, strict=True):
            args = (work_remote, remote, CloudpickleWrapper(env_fns_chunk))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        super().__init__(len(env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        assert len(actions) == self.num_envs, "Number of actions must match number of environments"
        for remote, inds in zip(self.remotes, self.env_inds_chunks, strict=True):
            if isinstance(actions, np.ndarray):
                remote.send(("step", actions[inds]))
            else:
                remote.send(("step", [actions[i] for i in inds]))
        self.waiting = True

    def _get_and_reorder_results(self, env_inds_chunks=None) -> list[Any]:
        if env_inds_chunks is None:
            env_inds_chunks = self.env_inds_chunks
            results = [None] * self.num_envs
        else:
            results = [None] * sum([len(chunk) for chunk in env_inds_chunks])
        for remote, env_inds_chunk in zip(self.remotes, env_inds_chunks, strict=True):
            if len(env_inds_chunk) == 0:
                continue
            result = remote.recv()
            for ind, env_ind in enumerate(env_inds_chunk):
                results[env_ind] = result[ind]
        return results

    def step_wait(self) -> VecEnvStepReturn:
        results = self._get_and_reorder_results()
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results, strict=True)  # type: ignore[assignment]
        return _stack_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos  # type: ignore[return-value]

    def reset(self) -> VecEnvObs:
        for worker_ind, env_inds_chunk in enumerate(self.env_inds_chunks):
            self.remotes[worker_ind].send((
                "reset",
                [{
                    "seed": self._seeds[env_ind],
                    "options": self._options[env_ind],
                } if self._options[env_ind] else {
                    "seed": self._seeds[env_ind],
                }
                for env_ind in env_inds_chunk],
            ))  # fmt: skip
        results = self._get_and_reorder_results()
        obs, self.reset_infos = zip(*results, strict=True)  # type: ignore[assignment]
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return _stack_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray | None]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remotes]
        for pipe in self.remotes:
            # gather render return from subprocesses
            pipe.send(("render", None))
        outputs = self._get_and_reorder_results()
        return outputs

    def has_attr(self, attr_name: str) -> bool:
        """Check if an attribute exists for a vectorized environment. (see base class)."""
        for remote in self.remotes:
            remote.send(("has_attr", attr_name))
        result = self._get_and_reorder_results()
        return all(result)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """Return attribute from vectorized environment (see base class)."""
        worker_payloads, inds_chunks = self._indices_to_payloads(indices)
        for worker_ind, worker_payload in enumerate(worker_payloads):
            if worker_payload is None or len(worker_payload) > 0:
                self.remotes[worker_ind].send(("get_attr", [worker_payload, attr_name]))
        result = self._get_and_reorder_results(inds_chunks)
        return result

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        worker_payloads, inds_chunks = self._indices_to_payloads(indices)
        for worker_ind, worker_payload in enumerate(worker_payloads):
            if worker_payload is None or len(worker_payload) > 0:
                self.remotes[worker_ind].send(("set_attr", [worker_payload, attr_name, value]))
        result = self._get_and_reorder_results(inds_chunks)  # noqa: F841
        # return result
        return None

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        """Call instance methods of vectorized environments."""
        worker_payloads, inds_chunks = self._indices_to_payloads(indices)
        for worker_ind, worker_payload in enumerate(worker_payloads):
            if worker_payload is None or len(worker_payload) > 0:
                self.remotes[worker_ind].send(("env_method", [worker_payload, method_name, method_args, method_kwargs]))
        result = self._get_and_reorder_results(inds_chunks)
        return result

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        worker_payloads, inds_chunks = self._indices_to_payloads(indices)
        for worker_ind, worker_payload in enumerate(worker_payloads):
            if worker_payload is None or len(worker_payload) > 0:
                self.remotes[worker_ind].send(("is_wrapped", [worker_payload, wrapper_class]))
        result = self._get_and_reorder_results(inds_chunks)
        return result

    def _indices_to_payloads(self, indices: VecEnvIndices = None):
        if isinstance(indices, int):
            indices = [indices]
        if indices is None:
            worker_payloads = [None] * self.num_worker  # type: ignore[var-annotated]
            inds_chunks = None  # type: ignore[var-annotated]
        else:  # sequence
            indices = [ind if ind >= 0 else self.num_envs + ind for ind in indices]  # for ind == -1, -2, etc.
            worker_payloads = [[] for _ in range(self.num_worker)]  # type: ignore[misc]
            inds_chunks = [[] for _ in range(self.num_worker)]
            for i, ind in enumerate(indices):
                worker_indpair = self.env_ind_to_worker_indpair(ind)
                worker_payloads[worker_indpair[0]].append(worker_indpair[1])
                inds_chunks[worker_indpair[0]].append(i)
        return worker_payloads, inds_chunks


def _stack_obs(obs_list: list[VecEnvObs] | tuple[VecEnvObs], space: spaces.Space) -> VecEnvObs:
    """
    Stack observations (convert from a list of single env obs to a stack of obs),
    depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: Concatenated observations.
            A NumPy array or a dict or tuple of stacked numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs_list, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs_list) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, dict), "Dict space must have ordered subspaces"
        assert isinstance(obs_list[0], dict), "non-dict observation for environment with Dict observation space"
        return {key: np.stack([single_obs[key] for single_obs in obs_list]) for key in space.spaces.keys()}  # type: ignore[call-overload]
    elif isinstance(space, spaces.Tuple):
        assert isinstance(obs_list[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([single_obs[i] for single_obs in obs_list]) for i in range(obs_len))  # type: ignore[index]
    else:
        return np.stack(obs_list)  # type: ignore[arg-type]
