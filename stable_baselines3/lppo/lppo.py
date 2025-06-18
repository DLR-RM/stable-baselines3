import io
import pathlib
import sys
import time
from collections import deque
from typing import Union, Optional, Any, List

import numpy as np

import torch as th
from stable_baselines3.common.vec_env.patch_gym import _convert_space

from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr

from stable_baselines3.common.utils import get_system_info, check_for_correct_spaces, safe_mean

from stable_baselines3.common.type_aliases import GymEnv, Schedule

from stable_baselines3.common.base_class import SelfBaseAlgorithm
from torch.nn import functional as F

from gymnasium import spaces

from stable_baselines3.lppo.policies import MoActorCriticPolicy
from stable_baselines3.ppo.ppo import PPO
import warnings


class LPPO(PPO):
    policy_aliases = {
        "MoMlpPolicy": MoActorCriticPolicy,
    }

    def __init__(self, policy, env, n_objectives, eta_values : List[float]=None, beta_values : List[float]=None, tolerance : Union[float, Schedule] = 3e-5, *args, **kwargs):
        super().__init__(policy, env, *args, **kwargs)
        # We need to replace the default rollout buffer for the multi-objective one
        """self.rollout_buffer = MultiObjectiveRolloutBuffer(
            self.n_steps, self.observation_space, self.action_space, self.device, n_objectives=n_objectives,
            n_envs=env.num_envs, **self.rollout_buffer_kwargs
        )"""
        self.eta_values = np.array(eta_values).reshape((-1))

        self.beta_values = np.array(beta_values)
        self.mu_values = np.array([0.0] * (n_objectives - 1))
        self.n_objectives = n_objectives
        self.recent_losses = [deque(maxlen=50) for _ in range(self.n_objectives)]
        self.tolerance = tolerance
        self.j = np.zeros(self.n_objectives - 1)

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # Lexico update
        first_order_weights = th.zeros(self.n_objectives).to(self.policy.device)
        for obj in range(self.n_objectives - 1):
            w = self.beta_values[obj] + self.mu_values[obj] * sum(
                [self.beta_values[j] for j in range(obj + 1, self.n_objectives)])
            first_order_weights[obj] = w
        first_order_weights[-1] = th.tensor(self.beta_values[self.n_objectives - 1],
                                            dtype=first_order_weights.dtype)
        last_values = None
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.reshape(-1, self.n_objectives)
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    # normalize advantages per objective
                    advantages = (advantages - advantages.mean(axis=0)) / (advantages.std(axis=0) + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                first_order_weighted_advantages = th.matmul(advantages, first_order_weights)

                policy_loss_1 = first_order_weighted_advantages * ratio
                policy_loss_2 = first_order_weighted_advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Keep track of the losses independently for each agent and objective
                for obj in range(self.n_objectives):
                    _surr1 = ratio * advantages[:, obj]
                    _surr2 = (
                            th.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages[:, obj]
                    )
                    _surr = th.min(_surr1, _surr2)
                    _pg_loss = -((_surr + self.ent_coef * entropy)).mean()
                    self.recent_losses[obj].append(_pg_loss.detach().cpu())

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                unclipped_loss = F.mse_loss(rollout_data.returns, values)

                if self.clip_range_vf is None:
                    # No clipping
                    #values_pred = values
                    value_loss = unclipped_loss
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    if epoch > 0:
                        clipped_loss = F.mse_loss(
                            rollout_data.returns,
                            th.clamp(
                                values,
                                last_values - clip_range_vf,
                                last_values + clip_range_vf,
                            ))
                        value_loss = th.min(unclipped_loss, clipped_loss)
                    else:
                        # First iteration, no last_values
                        value_loss = unclipped_loss
                # Value loss using the TD(gae_lambda) target
                #value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                last_values = values.detach()
                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # Update Lagrange multipliers
        tol = self.tolerance if isinstance(self.tolerance, float) else self.tolerance(self._current_progress_remaining)
        for i in range(self.n_objectives - 1):
            self.j[i] = (-th.tensor(self.recent_losses[i])).mean()
            self.mu_values[i] += self.eta_values[i] * (self.j[i] - tol - (-self.recent_losses[i][-1]) )
            self.mu_values[i] = max(0, self.mu_values[i])

        #explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/ent_coef", self.ent_coef)

        for obj in range(self.n_objectives):
            self.logger.record(f"train_mo/advantage_scalarisation_weight_{obj}", first_order_weights[obj].float().item())
            self.logger.record(f"train_mo/loss_{obj}", np.mean(self.recent_losses[obj]))
        for obj in range(self.n_objectives-1):
            self.logger.record(f"train_mo/mu_{obj}", self.mu_values[obj])
        self.logger.record(f"train_mo/tolerance", tol)
        #self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    @classmethod
    def load(  # noqa: C901
            cls: type[SelfBaseAlgorithm],
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            env: Optional[GymEnv] = None,
            device: Union[th.device, str] = "auto",
            custom_objects: Optional[dict[str, Any]] = None,
            print_system_info: bool = False,
            force_reset: bool = True,
            **kwargs,
    ) -> SelfBaseAlgorithm:

        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            saved_net_arch = data["policy_kwargs"].get("net_arch")
            if saved_net_arch and isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
            n_objectives=data["n_objectives"],
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load policies saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a A2C/PPO model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        except ValueError as e:
            # Patch to load DQN policies saved using SB3 < 2.4.0
            # The target network params are no longer in the optimizer
            # See https://github.com/DLR-RM/stable-baselines3/pull/1963
            saved_optim_params = params["policy.optimizer"]["param_groups"][0]["params"]  # type: ignore[index]
            n_params_saved = len(saved_optim_params)
            n_params = len(model.policy.optimizer.param_groups[0]["params"])
            if n_params_saved == 2 * n_params:
                # Truncate to include only online network params
                params["policy.optimizer"]["param_groups"][0]["params"] = saved_optim_params[
                                                                          :n_params]  # type: ignore[index]

                model.set_parameters(params, exact_match=True, device=device)
                warnings.warn(
                    "You are probably loading a DQN model saved with SB3 < 2.4.0, "
                    "we truncated the optimizer state so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/pull/1963 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
        return model

    def dump_logs(self, iteration: int = 0) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            for m in range(self.n_objectives):
                self.logger.record(f"rollout/ep_rew_mean_r{m}", safe_mean([ep_info[f"r{m}"] for ep_info in self.ep_info_buffer]))

            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)
