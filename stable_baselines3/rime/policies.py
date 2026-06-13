"""RIME-PPO Policy: Actor-Critic policy with RNA-editing inspired online adaptation.

This module implements the RIMEMlpPolicy, which extends ActorCriticPolicy with:
- Dual-signal driven editing rate (alpha_t)
- Low-rank multiplicative modulation injection (Delta_h_t)
- Critic with stop_gradient on e_t to prevent gradient conflict
"""

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule


class RIMEMlpPolicy(ActorCriticPolicy):
    """RIME-PPO Policy with RNA-editing inspired online adaptation.

    Key innovations over standard ActorCriticPolicy:
    1. Editing state e_t: fast variable updated within each episode
    2. Dual-signal editing rate alpha_t: driven by surprise (psi_t) and reward rate (rho_t)
    3. Low-rank multiplicative modulation: Delta_h_t = U(tanh(e_t) * LayerNorm(V*h_t))
    4. Critic receives stop_gradient(e_t) to prevent gradient conflict

    Default hyperparameters:
    - S=32 (editing dimension)
    - d=256 (hidden dimension)
    - eps_max=0.15 (max editing magnitude)
    - a0=-2 (initial bias for editing rate, so alpha starts small)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        # RIME-specific hyperparameters
        rime_edit_dim: int = 32,  # S: editing state dimension
        rime_hidden_dim: int = 256,  # d: hidden dimension for policy/value
        rime_eps_max: float = 0.15,  # max editing magnitude
        rime_a0: float = -2.0,  # initial bias for editing rate
        rime_ema_decay: float = 0.99,  # EMA decay for obs tracking
        # Ablation flags
        rime_no_surprise: bool = False,  # disable psi_t (RIME-NoSurp)
        rime_no_reward_rate: bool = False,  # fix alpha_t (RIME-NoRewRate)
        rime_no_critic_e: bool = False,  # Critic without e_t (RIME-NoCriticE)
        rime_additive_injection: bool = False,  # additive instead of multiplicative (RIME-AddInj)
        rime_no_layer_norm: bool = False,  # remove LayerNorm (RIME-NoLN)
        # Standard policy args
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        # Use ReLU activation and custom net_arch by default
        if net_arch is None:
            net_arch = dict(pi=[rime_hidden_dim], vf=[rime_hidden_dim])

        self.rime_edit_dim = rime_edit_dim
        self.rime_hidden_dim = rime_hidden_dim
        self.rime_eps_max = rime_eps_max
        self.rime_a0 = rime_a0
        self.rime_ema_decay = rime_ema_decay

        # Ablation flags
        self.rime_no_surprise = rime_no_surprise
        self.rime_no_reward_rate = rime_no_reward_rate
        self.rime_no_critic_e = rime_no_critic_e
        self.rime_additive_injection = rime_additive_injection
        self.rime_no_layer_norm = rime_no_layer_norm

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """Build all RIME-specific networks in addition to standard policy."""
        # Build standard MLP extractor first
        super()._build(lr_schedule)

        obs_dim = self.features_dim
        S = self.rime_edit_dim
        d = self.mlp_extractor.latent_dim_pi  # should match rime_hidden_dim
        act_dim = get_action_dim(self.action_space)

        # --- Low-rank multiplicative modulation matrices ---
        # U: (d, S), V: (S, d) — low-rank decomposition
        self.U = nn.Linear(S, d, bias=False)
        self.V = nn.Linear(d, S, bias=False)
        # Small initialization for stability
        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.V.weight, std=0.01)

        # LayerNorm for V*h_t (critical for numerical stability)
        if not self.rime_no_layer_norm:
            self.edit_layer_norm = nn.LayerNorm(S)
        else:
            self.edit_layer_norm = None

        # --- Editing rate parameters ---
        # alpha_t = sigma(a0 + beta_s * psi_t + beta_r * rho_t)
        self.a0 = nn.Parameter(th.tensor(self.rime_a0))
        self.log_beta_s = nn.Parameter(th.tensor(0.0))  # log-space for positivity
        self.log_beta_r = nn.Parameter(th.tensor(0.0))

        # --- Context network for computing editing target m_t ---
        # c_t = [rho_t; W_proj(o_t - obs_ema); a_{t-1}]
        # W_proj maps obs difference to S dimensions
        self.W_proj = nn.Linear(obs_dim, S, bias=False)
        # Context net: maps concatenated context to hidden
        ctx_input_dim = 1 + S + act_dim  # rho_t + W_proj(delta_obs) + prev_action
        self.ctx_net = nn.Linear(ctx_input_dim, S)
        # W_m: maps context to editing target
        self.W_m = nn.Linear(S, S, bias=False)

        # --- Critic projection for e_t ---
        # Critic input: [obs_features; W_v_proj(e_t.detach())]
        if not self.rime_no_critic_e:
            self.W_v_proj = nn.Linear(S, S, bias=False)
            # Replace value_net to accept extended input
            vf_latent_dim = self.mlp_extractor.latent_dim_vf
            self.value_net = nn.Linear(vf_latent_dim + S, 1)
        else:
            self.W_v_proj = None

    def forward_rime(
        self,
        obs: th.Tensor,
        e_t: th.Tensor,
        obs_ema: th.Tensor,
        prev_action: th.Tensor,
        rho_t: th.Tensor,
        psi_t: th.Tensor,
        done: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Complete RIME forward pass with editing mechanism.

        Args:
            obs: Current observation (n_envs, obs_dim).
            e_t: Current editing state (n_envs, S).
            obs_ema: EMA of observations (n_envs, obs_dim).
            prev_action: Previous action (n_envs, act_dim).
            rho_t: Standardized TD error signal (n_envs,).
            psi_t: Surprise signal (n_envs,).
            done: Episode done flag (n_envs,).
            deterministic: Whether to use deterministic actions.

        Returns:
            actions: Selected actions (n_envs, act_dim).
            values: State values (n_envs, 1).
            log_probs: Log probabilities of actions (n_envs,).
            e_next: Next editing state (n_envs, S).
            alpha_t: Editing rate (n_envs,).
        """
        # Reset e_t at episode boundaries
        mask = (1.0 - done).unsqueeze(-1)  # (n_envs, 1)
        e_t = e_t * mask

        # Ensure float32 dtype for all RIME computations
        obs_ema_f = obs_ema.float()
        prev_action_f = prev_action.float()
        rho_t_f = rho_t.float()
        psi_t_f = psi_t.float()

        # --- Compute editing rate alpha_t ---
        beta_s = th.exp(self.log_beta_s)
        beta_r = th.exp(self.log_beta_r)

        if self.rime_no_surprise:
            psi_input = th.zeros_like(psi_t_f)
        else:
            psi_input = psi_t_f

        if self.rime_no_reward_rate:
            # Fixed alpha (no reward rate modulation)
            alpha_t = th.sigmoid(self.a0 * th.ones_like(psi_t_f))
        else:
            alpha_t = th.sigmoid(self.a0 + beta_s * psi_input + beta_r * rho_t_f)

        # --- Compute editing target m_t ---
        delta_obs = obs.float() - obs_ema_f  # (n_envs, obs_dim)
        proj_delta = self.W_proj(delta_obs)  # (n_envs, S)

        # Context: [rho_t; W_proj(delta_obs); prev_action]
        rho_expanded = rho_t_f.unsqueeze(-1)  # (n_envs, 1)
        ctx = th.cat([rho_expanded, proj_delta, prev_action_f], dim=-1)  # (n_envs, 1+S+act_dim)
        ctx_hidden = th.relu(self.ctx_net(ctx))  # (n_envs, S)
        m_t = self.rime_eps_max * th.tanh(self.W_m(ctx_hidden))  # (n_envs, S)

        # --- Update editing state ---
        alpha_expanded = alpha_t.unsqueeze(-1)  # (n_envs, 1)
        e_next = (1 - alpha_expanded) * e_t + alpha_expanded * m_t  # (n_envs, S)

        # --- Policy forward with editing modulation ---
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # --- Apply low-rank multiplicative modulation to policy latent ---
        Vh = self.V(latent_pi)  # (n_envs, S)
        if self.edit_layer_norm is not None:
            Vh = self.edit_layer_norm(Vh)

        if self.rime_additive_injection:
            # Additive injection (ablation: RIME-AddInj)
            delta_h = self.U(th.tanh(e_next) * Vh)
        else:
            # Multiplicative modulation (default)
            delta_h = self.U(th.tanh(e_next) * Vh)

        latent_pi_mod = latent_pi + delta_h  # (n_envs, d)

        # --- Action distribution ---
        distribution = self._get_action_dist_from_latent(latent_pi_mod)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        # --- Critic with stop_gradient(e_t) ---
        if not self.rime_no_critic_e and self.W_v_proj is not None:
            e_for_critic = self.W_v_proj(e_next.detach())  # stop_gradient via .detach()
            values = self.value_net(th.cat([latent_vf, e_for_critic], dim=-1))
        else:
            values = self.value_net(latent_vf)

        return actions, values, log_prob, e_next, alpha_t

    def evaluate_actions_rime(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        e_t: th.Tensor,
        obs_ema: th.Tensor,
        prev_action: th.Tensor,
        rho_t: th.Tensor,
        psi_t: th.Tensor,
        done: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None, th.Tensor, th.Tensor]:
        """Evaluate actions for training with RIME mechanism (BPTT).

        Same as forward_rime but returns values, log_prob, entropy for PPO loss computation.
        """

        # Reset e_t at episode boundaries
        mask = (1.0 - done).unsqueeze(-1)
        e_t = e_t * mask

        # Ensure float32 dtype for all RIME computations
        obs_ema_f = obs_ema.float()
        prev_action_f = prev_action.float()
        rho_t_f = rho_t.float()
        psi_t_f = psi_t.float()

        # --- Compute editing rate alpha_t ---
        beta_s = th.exp(self.log_beta_s)
        beta_r = th.exp(self.log_beta_r)

        if self.rime_no_surprise:
            psi_input = th.zeros_like(psi_t_f)
        else:
            psi_input = psi_t_f

        if self.rime_no_reward_rate:
            alpha_t = th.sigmoid(self.a0 * th.ones_like(psi_t_f))
        else:
            alpha_t = th.sigmoid(self.a0 + beta_s * psi_input + beta_r * rho_t_f)

        # --- Compute editing target m_t ---
        delta_obs = obs.float() - obs_ema_f
        proj_delta = self.W_proj(delta_obs)

        rho_expanded = rho_t_f.unsqueeze(-1)
        ctx = th.cat([rho_expanded, proj_delta, prev_action_f], dim=-1)
        ctx_hidden = th.relu(self.ctx_net(ctx))
        m_t = self.rime_eps_max * th.tanh(self.W_m(ctx_hidden))

        # --- Update editing state ---
        alpha_expanded = alpha_t.unsqueeze(-1)
        e_next = (1 - alpha_expanded) * e_t + alpha_expanded * m_t

        # --- Policy forward with editing modulation ---
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # --- Apply low-rank multiplicative modulation ---
        Vh = self.V(latent_pi)
        if self.edit_layer_norm is not None:
            Vh = self.edit_layer_norm(Vh)

        if self.rime_additive_injection:
            delta_h = self.U(th.tanh(e_next) * Vh)
        else:
            delta_h = self.U(th.tanh(e_next) * Vh)

        latent_pi_mod = latent_pi + delta_h

        # --- Action distribution ---
        distribution = self._get_action_dist_from_latent(latent_pi_mod)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # --- Critic with stop_gradient(e_t) ---
        if not self.rime_no_critic_e and self.W_v_proj is not None:
            e_for_critic = self.W_v_proj(e_next.detach())
            values = self.value_net(th.cat([latent_vf, e_for_critic], dim=-1))
        else:
            values = self.value_net(latent_vf)

        return values, log_prob, entropy, e_next, alpha_t

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Standard forward pass (for compatibility with SB3 predict).

        Uses zero e_t (no editing) for standard behavior.
        """
        # Fallback: standard forward without editing
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        if not self.rime_no_critic_e and self.W_v_proj is not None:
            batch_size = latent_vf.shape[0]
            zero_e = th.zeros(batch_size, self.rime_edit_dim, device=latent_vf.device)
            e_for_critic = self.W_v_proj(zero_e)
            values = self.value_net(th.cat([latent_vf, e_for_critic], dim=-1))
        else:
            values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """Get estimated values. When Critic uses e_t, pass zero e_t (no editing context)."""
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        if not self.rime_no_critic_e and self.W_v_proj is not None:
            batch_size = latent_vf.shape[0]
            zero_e = th.zeros(batch_size, self.rime_edit_dim, device=latent_vf.device)
            e_for_critic = self.W_v_proj(zero_e)
            return self.value_net(th.cat([latent_vf, e_for_critic], dim=-1))
        return self.value_net(latent_vf)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        """Standard evaluate_actions (for compatibility).

        Note: RIME-PPO uses evaluate_actions_rime instead during training.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        if not self.rime_no_critic_e and self.W_v_proj is not None:
            batch_size = latent_vf.shape[0]
            zero_e = th.zeros(batch_size, self.rime_edit_dim, device=latent_vf.device)
            e_for_critic = self.W_v_proj(zero_e)
            values = self.value_net(th.cat([latent_vf, e_for_critic], dim=-1))
        else:
            values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            rime_edit_dim=self.rime_edit_dim,
            rime_hidden_dim=self.rime_hidden_dim,
            rime_eps_max=self.rime_eps_max,
            rime_a0=self.rime_a0,
            rime_ema_decay=self.rime_ema_decay,
            rime_no_surprise=self.rime_no_surprise,
            rime_no_reward_rate=self.rime_no_reward_rate,
            rime_no_critic_e=self.rime_no_critic_e,
            rime_additive_injection=self.rime_additive_injection,
            rime_no_layer_norm=self.rime_no_layer_norm,
        )
        return data


def get_action_dim(action_space: spaces.Space) -> int:
    """Get the dimension of the action space."""
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        return len(action_space.nvec)
    elif isinstance(action_space, spaces.MultiBinary):
        return int(action_space.n)
    else:
        raise ValueError(f"Unsupported action space: {action_space}")
