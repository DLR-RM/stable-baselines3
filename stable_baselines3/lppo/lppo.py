from collections import deque
import numpy as np

import torch as th
from torch.nn import functional as F

from gymnasium import spaces

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.buffers import MoRolloutBuffer
import warnings
# TODO: Add support for a tolerance parameter.
class LPPO(PPO):
    """
    LPPO (Lexicographic Proximal Policy Optimization) implementation. Multi-objective extension of PPO that uses lagrangian
    multipliers to optimize the policy with respect to multiple ordered objectives.
    Unofficial implementation code of the original paper about LPPO: https://arxiv.org/pdf/2212.13769

    :param policy: (ActorCriticPolicy or str) The policy model to use (MoMlpPolicy). Value function must have the same number of outputs as the number of objectives.
    :param env: (Gym Environment) The environment to learn from. To use parallel envs, use MoSubprocVecEnv.
    :param n_objectives: (int) The number of objectives to optimize.
    :param eta_values: (list of float) The learning rate for the lagrangian multipliers. See paper for more details.
    :param beta_values: (list of float) Coefficients that establish the priorisation of the objectives. Higher values means higher ranking. See paper for more details.
    """
    def __init__(self, policy, env, n_objectives, eta_values=None, beta_values=None, *args, **kwargs):
        super().__init__(policy, env, *args, **kwargs)
        # We need to replace the default rollout buffer for the multi-objective one
        self.rollout_buffer = MoRolloutBuffer(
            self.n_steps, self.observation_space, self.action_space, self.device, n_objectives=n_objectives,
            n_envs=env.num_envs, **self.rollout_buffer_kwargs
        )

        if beta_values is None:
            warnings.warn("Beta values not provided. Assuming they are ordered in descending priority.")
            beta_values = [i+1 for i in range(n_objectives)]
            beta_values.reverse()

        if eta_values is None:
            warnings.warn("Eta values not provided. Assuming they are all 0.1.")
            eta_values = [0.1 for _ in range(n_objectives - 1)]

        self.beta_values = np.array(beta_values)
        self.eta_values = np.array(eta_values).reshape((-1))
        self.mu_values = np.array([0.0] * (n_objectives - 1))
        self.n_objectives = n_objectives
        self.recent_losses = [deque(maxlen=50) for _ in range(self.n_objectives)]
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
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
                    _pg_loss = -((_surr1 + self.ent_coef * entropy)).sum()
                    self.recent_losses[obj].append(_pg_loss.detach())

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

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
        for i in range(self.n_objectives - 1):
            self.j[i] = (-th.tensor(self.recent_losses[i])).mean()
            self.mu_values[i] += self.eta_values[i] * (self.j[i] - (-self.recent_losses[i][-1]))
            self.mu_values[i] = max(0, self.mu_values[i])
        #explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/mu", self.mu_values)
        self.logger.record("train/advantage_scalarisation_weights", first_order_weights)
        #self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

