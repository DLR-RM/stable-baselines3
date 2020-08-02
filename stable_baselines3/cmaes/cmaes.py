import time
from typing import Any, Dict, Optional, Type, Union

import cma
import numpy as np
import torch as th

from stable_baselines3.cmaes.policies import CMAESPolicy
from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.sac import SAC


class CMAES(BaseAlgorithm):
    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        n_steps: int = 200,
        n_individuals: int = -1,
        std_init: float = 0.5,
        best_individual: Union[np.ndarray, None, str] = None,
        diagonal_cov: bool = False,
        max_hist: int = 10,
        pop_size: Optional[int] = None,
        policy_kwargs: Dict[str, Any] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
    ):

        super(CMAES, self).__init__(
            policy=policy,
            env=env,
            policy_base=CMAESPolicy,
            learning_rate=0.0,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=True,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=False,
        )

        self.policy_kwargs["device"] = self.device
        if isinstance(best_individual, str):
            best_individual = SAC.load(best_individual).actor.parameters_to_vector()  # pytype:disable=attribute-error
        self.best_individual = best_individual
        self.best_ever = None
        self.std_init = std_init
        self.n_steps = n_steps
        self.es = None
        self.diagonal_cov = diagonal_cov
        self.pop_size = pop_size
        self.max_hist = max_hist
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(
            self.observation_space, self.action_space, **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        self.actor = self.policy.actor

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 10,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "CMAES":

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        if self.best_individual is None:
            self.best_individual = self.actor.parameters_to_vector()

        if self.best_ever is None:
            self.best_ever = cma.optimization_tools.BestSolution()

        if self.es is None:
            options = {"seed": self.seed}
            if self.env.num_envs > 1:
                options["popsize"] = self.env.num_envs
            if self.pop_size is not None:
                options["popsize"] = self.pop_size
            if self.diagonal_cov:
                options["CMA_diagonal"] = True
            self.es = cma.CMAEvolutionStrategy(self.best_individual, self.std_init, options)

        continue_training = True

        while self.num_timesteps < total_timesteps and not self.es.stop() and continue_training:
            candidates = self.es.ask()

            # Prevent high memory usage but changes `es.stop()` behavior
            if len(self.es.fit.hist) > self.max_hist:
                try:
                    self.es.fit.hist.pop()
                    self.es.fit.histbest.pop()
                    self.es.fit.histmedian.pop()
                except IndexError:
                    # Removing element from empty list
                    pass

            # Add best
            candidates.append(self.best_individual)
            returns = np.zeros((len(candidates),))
            candidate_idx = 0
            candidate_steps = 0

            self.actor.load_from_vector(candidates[candidate_idx])
            callback.on_rollout_start()

            while candidate_idx < len(candidates):
                # TODO support num_envs > 0
                action, _ = self.actor.predict(self._last_obs, deterministic=True)

                # Rescale and perform action
                new_obs, reward, done, infos = self.env.step(action)

                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    continue_training = False
                    break

                returns[candidate_idx] += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                self._last_obs = new_obs

                self.num_timesteps += 1
                candidate_steps += 1
                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                if candidate_steps > self.n_steps:
                    if self.verbose > 0:
                        print(f"Candidate {candidate_idx + 1}, return={returns[candidate_idx]:.2f}")
                    # force reset
                    self._last_obs = self.env.reset()
                    candidate_idx += 1
                    candidate_steps = 0
                    if candidate_idx < len(candidates):
                        self.actor.load_from_vector(candidates[candidate_idx])
                    else:
                        break

                if done:
                    self._episode_num += 1

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

            callback.on_rollout_end()

            # TODO: inject best solution from time to time?
            self.es.tell(candidates, -1 * returns)
            if self.verbose > 0:
                print(f"Mean return={np.mean(returns):.2f} +/- {np.std(returns):.2f}")
            # TODO: load best individual when using predict
            self.best_ever.update(self.es.best)
            self.best_individual = self.best_ever.x
            self.policy.best_actor.load_from_vector(self.best_individual)

        # TODO: fix saving/loading
        # del self.es
        callback.on_training_end()

        return self

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        fps = int(self.num_timesteps / (time.time() - self.start_time))
        logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        logger.record("time/fps", fps)
        logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
        logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
        logger.record("rollout/best_ever", -self.best_ever.f)

        if len(self.ep_success_buffer) > 0:
            logger.record("rollout/success rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        logger.dump(step=self.num_timesteps)
