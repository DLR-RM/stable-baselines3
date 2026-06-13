"""Utility classes for RIME-PPO: WelfordOnline standardizer and ObsEMA tracker."""

import torch as th


class WelfordOnline:
    """Online standardization of TD errors using Welford's algorithm.

    Maintains running mean and variance, and provides standardized (z-score)
    values with detach to prevent gradient flow through the statistics.
    """

    def __init__(self, device: th.device | str = "cpu"):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.device = device

    def update_and_standardize(self, delta: th.Tensor, clip_val: float = 5.0) -> th.Tensor:
        """Update statistics with new TD error delta and return standardized rho_t.

        Args:
            delta: TD error tensor (detached before use).
            clip_val: Clipping range for standardized value.

        Returns:
            rho_t: Standardized and clipped TD error (detached).
        """
        # Detach to prevent gradient flow through statistics
        delta_detached = delta.detach()

        # Welford online update (scalar statistics)
        val = delta_detached.item()
        self.n += 1
        delta_w = val - self.mean
        self.mean += delta_w / self.n
        delta2 = val - self.mean
        self.m2 += delta_w * delta2

        # Compute std
        if self.n < 2:
            std = 1.0
        else:
            variance = self.m2 / (self.n - 1)
            std = max(variance ** 0.5, 1e-8)

        # Standardize and clip
        rho = (delta_detached - self.mean) / (std + 1e-8)
        rho = th.clamp(rho, -clip_val, clip_val)
        return rho

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0


class ObsEMA:
    """Exponential Moving Average of observations for computing psi_t (surprise signal).

    Maintains an EMA of the L2 norm of observation differences, used as
    the normalization factor for the environment change signal.
    """

    def __init__(self, obs_dim: int, n_envs: int = 1, ema_decay: float = 0.99, device: th.device | str = "cpu"):
        self.ema_decay = ema_decay
        self.device = device
        # Running EMA of ||delta_o||_2
        self.ema_norm = th.zeros(n_envs, device=device)
        self.obs_ema = th.zeros(n_envs, obs_dim, device=device)
        self.initialized = False

    def compute_psi(self, obs: th.Tensor, eps: float = 1e-8) -> th.Tensor:
        """Compute the surprise signal psi_t.

        Args:
            obs: Current observation tensor of shape (n_envs, obs_dim).

        Returns:
            psi_t: Surprise signal of shape (n_envs,).
        """
        if not self.initialized:
            self.obs_ema = obs.clone().detach()
            self.ema_norm = th.ones(obs.shape[0], device=self.device)
            self.initialized = True
            return th.zeros(obs.shape[0], device=self.device)

        # L2 norm of observation change
        delta_obs = obs - self.obs_ema
        delta_norm = th.norm(delta_obs, p=2, dim=-1)  # (n_envs,)

        # Update EMA of the norm
        self.ema_norm = self.ema_decay * self.ema_norm + (1 - self.ema_decay) * delta_norm.detach()

        # Compute psi_t (normalized surprise)
        psi = delta_norm / (self.ema_norm + eps)

        # Update obs EMA
        self.obs_ema = self.ema_decay * self.obs_ema + (1 - self.ema_decay) * obs.clone().detach()

        return psi

    def reset(self, obs: th.Tensor | None = None) -> None:
        """Reset EMA state, optionally with initial observation."""
        if obs is not None:
            self.obs_ema = obs.clone().detach()
            self.ema_norm = th.ones(obs.shape[0], device=self.device)
            self.initialized = True
        else:
            self.obs_ema.zero_()
            self.ema_norm.zero_()
            self.initialized = False
