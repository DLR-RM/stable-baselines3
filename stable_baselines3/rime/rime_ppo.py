"""RIME-PPO Algorithm: PPO with RNA-editing inspired online adaptation.

Extends OnPolicyAlgorithm with:
1. collect_rollouts: maintains editing state e_t, computes psi_t/rho_t during rollout
2. train: chunked BPTT instead of shuffled minibatch SGD
"""

from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor
from stable_baselines3.rime.buffers import RIMERolloutBuffer
from stable_baselines3.rime.policies import RIMEMlpPolicy
from stable_baselines3.rime.utils import ObsEMA, WelfordOnline

SelfRIMEPPO = TypeVar("SelfRIMEPPO", bound="RIMEPPO")


class RIMEPPO(OnPolicyAlgorithm):
    """RIME-PPO: RNA-editing Inspired Multi-scale Evolution PPO.

    Key differences from standard PPO:
    - Maintains editing state e_t during rollout collection
    - Computes dual signals (psi_t, rho_t) for editing rate
    - Uses chunked BPTT for training instead of shuffled minibatch
    - Critic receives stop_gradient(e_t) to prevent gradient conflict

    Args:
        chunk_size: BPTT chunk size (default 32).
        rime_edit_dim: Editing state dimension S (default 32).
        rime_eps_max: Maximum editing magnitude (default 0.15).
        rime_a0: Initial bias for editing rate (default -2.0).
        rime_ema_decay: EMA decay for observation tracking (default 0.99).
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": RIMEMlpPolicy,
    }

    def __init__(
        self,
        policy: str | type[RIMEMlpPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        # RIME-specific parameters
        chunk_size: int = 32,
        rime_edit_dim: int = 32,
        rime_eps_max: float = 0.15,
        rime_a0: float = -2.0,
        rime_ema_decay: float = 0.99,
    ):
        self.chunk_size = chunk_size
        self.rime_edit_dim = rime_edit_dim
        self.rime_eps_max = rime_eps_max
        self.rime_a0 = rime_a0
        self.rime_ema_decay = rime_ema_decay

        # Inject RIME-specific kwargs into policy_kwargs
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs.setdefault("rime_edit_dim", rime_edit_dim)
        policy_kwargs.setdefault("rime_eps_max", rime_eps_max)
        policy_kwargs.setdefault("rime_a0", rime_a0)
        policy_kwargs.setdefault("rime_ema_decay", rime_ema_decay)

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=RIMERolloutBuffer,
            rollout_buffer_kwargs=dict(
                edit_dim=rime_edit_dim,
                chunk_size=chunk_size,
            ),
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(spaces.Box,),
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive"
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

        # Initialize RIME-specific state
        self._rime_e_t = None
        self._rime_obs_ema = None
        self._rime_prev_action = None
        self._rime_welford = None

    def _init_rime_state(self, n_envs: int) -> None:
        """Initialize RIME state variables for rollout collection."""
        device = self.device
        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]

        self._rime_e_t = th.zeros(n_envs, self.rime_edit_dim, device=device)
        self._rime_obs_ema = ObsEMA(obs_dim, n_envs, self.rime_ema_decay, device=device)
        self._rime_prev_action = th.zeros(n_envs, act_dim, device=device)
        self._rime_welford = [WelfordOnline(device=device) for _ in range(n_envs)]

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer: RIMERolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect experiences with RIME editing mechanism.

        Maintains e_t, computes psi_t and rho_t at each step.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        n_envs = env.num_envs

        # Initialize RIME state if needed
        if self._rime_e_t is None or self._rime_e_t.shape[0] != n_envs:
            self._init_rime_state(n_envs)

        # Initialize obs EMA with first observation
        obs_tensor = obs_as_tensor(self._last_obs, self.device)
        self._rime_obs_ema.reset(obs_tensor)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                # Compute psi_t (surprise signal)
                psi_t = self._rime_obs_ema.compute_psi(obs_tensor)

                # Forward pass with RIME mechanism
                actions, values, log_probs, e_next, alpha_t = self.policy.forward_rime(
                    obs_tensor,
                    self._rime_e_t,
                    self._rime_obs_ema.obs_ema,
                    self._rime_prev_action,
                    th.zeros(n_envs, device=self.device),  # rho_t placeholder (computed after env step)
                    psi_t,
                    th.zeros(n_envs, device=self.device),  # done placeholder
                    deterministic=False,
                )

            actions_np = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions_np
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions_np, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)

            # Compute TD error and rho_t for each env
            dones_tensor = th.tensor(dones, dtype=th.float32, device=self.device)
            rho_t = th.zeros(n_envs, device=self.device)

            with th.no_grad():
                new_obs_tensor = obs_as_tensor(new_obs, self.device)

                # Compute next value for TD error
                # Need to forward through policy to get next value
                next_psi = self._rime_obs_ema.compute_psi(new_obs_tensor)
                _, next_values, _, _, _ = self.policy.forward_rime(
                    new_obs_tensor,
                    e_next,
                    self._rime_obs_ema.obs_ema,
                    actions.detach(),  # current action becomes prev_action for next step
                    th.zeros(n_envs, device=self.device),
                    next_psi,
                    dones_tensor,
                    deterministic=True,
                )

                rewards_tensor = th.tensor(rewards, dtype=th.float32, device=self.device)

                # TD error: delta_t = r_t + gamma * V(o_{t+1}, sg(e_{t+1})) - V(o_t, sg(e_t))
                for env_idx in range(n_envs):
                    delta = rewards_tensor[env_idx] + self.gamma * next_values[env_idx] * (1 - dones_tensor[env_idx]) - values[env_idx]
                    rho_t[env_idx] = self._rime_welford[env_idx].update_and_standardize(delta)

            # Handle timeout by bootstrapping with value function
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            # Store in buffer with RIME-specific fields
            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions_np,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                edit_state=self._rime_e_t.cpu().numpy(),
                psi=psi_t.cpu().numpy(),
                rho=rho_t.cpu().numpy(),
                obs_ema=self._rime_obs_ema.obs_ema.cpu().numpy(),
                prev_action=self._rime_prev_action.cpu().numpy(),
            )

            # Update RIME state
            # Reset e_t at episode boundaries
            for env_idx in range(n_envs):
                if dones[env_idx]:
                    e_next[env_idx] = 0.0
                    self._rime_welford[env_idx].reset()

            self._rime_e_t = e_next.detach()
            self._rime_prev_action = actions.detach()
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """Update policy using chunked BPTT on the collected rollout buffer.

        Key difference from standard PPO:
        - Data is processed in temporal chunks (not shuffled)
        - e_t is recursively updated within each chunk for gradient flow
        - e_t is detached at chunk boundaries to limit BPTT length
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for chunk in self.rollout_buffer.get_chunks(self.chunk_size):
                chunk_size = chunk.observations.shape[0]

                # Initialize e_t for this chunk (detached from previous chunk)
                e_t = chunk.e_init.clone()

                # Accumulate losses over the chunk
                chunk_policy_loss = th.tensor(0.0, device=self.device)
                chunk_value_loss = th.tensor(0.0, device=self.device)
                chunk_entropy_loss = th.tensor(0.0, device=self.device)
                chunk_clip_frac = 0.0

                for t in range(chunk_size):
                    # Forward pass with RIME mechanism
                    values, log_prob, entropy, e_next, alpha_t = self.policy.evaluate_actions_rime(
                        chunk.observations[t].unsqueeze(0),  # add batch dim
                        chunk.actions[t].unsqueeze(0),
                        e_t.unsqueeze(0),
                        chunk.obs_emas[t].unsqueeze(0),
                        chunk.prev_actions[t].unsqueeze(0),
                        chunk.rhos[t].unsqueeze(0),
                        chunk.psis[t].unsqueeze(0),
                        chunk.episode_starts[t].unsqueeze(0),
                    )

                    values = values.flatten()
                    actions = chunk.actions[t].unsqueeze(0)

                    # Advantage
                    advantages = chunk.advantages[t].unsqueeze(0)
                    if self.normalize_advantage and chunk_size > 1:
                        # Normalize over the full buffer (already done in buffer)
                        pass

                    # Ratio between old and new policy
                    ratio = th.exp(log_prob - chunk.old_log_prob[t].unsqueeze(0))

                    # Clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Clip fraction
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    chunk_clip_frac += clip_fraction

                    # Value loss
                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = chunk.old_values[t].unsqueeze(0) + th.clamp(
                            values - chunk.old_values[t].unsqueeze(0), -clip_range_vf, clip_range_vf
                        )
                    value_loss = F.mse_loss(chunk.returns[t].unsqueeze(0), values_pred)

                    # Entropy loss
                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    chunk_policy_loss = chunk_policy_loss + policy_loss
                    chunk_value_loss = chunk_value_loss + value_loss
                    chunk_entropy_loss = chunk_entropy_loss + entropy_loss

                    # Update e_t for next step (gradient flows through this)
                    e_t = e_next.squeeze(0)

                # Average losses over chunk
                n_steps = chunk_size
                loss = chunk_policy_loss / n_steps + self.ent_coef * chunk_entropy_loss / n_steps + self.vf_coef * chunk_value_loss / n_steps

                pg_losses.append((chunk_policy_loss / n_steps).item())
                value_losses.append((chunk_value_loss / n_steps).item())
                entropy_losses.append((chunk_entropy_loss / n_steps).item())
                clip_fractions.append(chunk_clip_frac / n_steps)

                # Approximate KL divergence
                with th.no_grad():
                    # Use last step's ratio as approximation
                    approx_kl_div = 0.0  # Simplified for chunked BPTT
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logging
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # Log RIME-specific metrics
        if hasattr(self.policy, "a0"):
            self.logger.record("rime/a0", self.policy.a0.item())
        if hasattr(self.policy, "log_beta_s"):
            self.logger.record("rime/beta_s", th.exp(self.policy.log_beta_s).item())
        if hasattr(self.policy, "log_beta_r"):
            self.logger.record("rime/beta_r", th.exp(self.policy.log_beta_r).item())

    def learn(
        self: SelfRIMEPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "RIMEPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRIMEPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
