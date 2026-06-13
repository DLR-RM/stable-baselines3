"""Non-stationary MuJoCo environment wrapper for RIME-PPO experiments.

Periodically changes dynamics parameters (mass, friction) to simulate
non-stationary environments, as required by the RIME-PPO paper.
"""

import gymnasium as gym
import numpy as np


class NonStationaryMuJoCoWrapper(gym.Wrapper):
    """Wrapper that introduces periodic dynamics changes in MuJoCo environments.

    Simulates non-stationary environments by modifying body mass and/or
    joint friction at regular intervals. This creates the sudden dynamics
    shifts that RIME-PPO is designed to handle.

    Args:
        env: The MuJoCo environment to wrap.
        change_interval: Number of steps between dynamics changes.
        mass_change_range: Relative range for mass changes, e.g. 0.5 means
            mass can change by ±50%.
        friction_change_range: Relative range for friction changes.
        change_both: Whether to change both mass and friction simultaneously.
        verbose: Whether to print when dynamics change.
    """

    def __init__(
        self,
        env: gym.Env,
        change_interval: int = 500,
        mass_change_range: float = 0.5,
        friction_change_range: float = 0.5,
        change_both: bool = True,
        verbose: bool = False,
    ):
        super().__init__(env)
        self.change_interval = change_interval
        self.mass_change_range = mass_change_range
        self.friction_change_range = friction_change_range
        self.change_both = change_both
        self.verbose = verbose

        self.step_count = 0
        self.change_count = 0

        # Store original dynamics parameters
        self._original_masses = None
        self._original_frictions = None
        self._store_original_params()

    def _store_original_params(self) -> None:
        """Store original model parameters for reference."""
        try:
            model = self.unwrapped.model
            self._original_masses = model.body_mass.copy()
            if hasattr(model, "geom_friction"):
                self._original_frictions = model.geom_friction.copy()
        except AttributeError:
            pass

    def _apply_dynamics_change(self) -> None:
        """Apply a random dynamics perturbation."""
        try:
            model = self.unwrapped.model

            if self._original_masses is not None:
                # Randomly perturb body masses
                n_bodies = len(self._original_masses)
                # Skip body 0 (world body)
                for i in range(1, n_bodies):
                    factor = 1.0 + np.random.uniform(
                        -self.mass_change_range, self.mass_change_range
                    )
                    model.body_mass[i] = self._original_masses[i] * factor

            if self.change_both and self._original_frictions is not None:
                # Randomly perturb friction
                n_geoms = len(self._original_frictions)
                for i in range(n_geoms):
                    factor = 1.0 + np.random.uniform(
                        -self.friction_change_range, self.friction_change_range
                    )
                    # Only change the first friction coefficient (slide)
                    model.geom_friction[i, 0] = max(
                        0.1, self._original_frictions[i, 0] * factor
                    )

            self.change_count += 1
            if self.verbose:
                print(f"[Step {self.step_count}] Dynamics change #{self.change_count}")

        except AttributeError:
            if self.verbose:
                print(f"[Step {self.step_count}] Warning: Could not modify dynamics")

    def reset(self, **kwargs):
        self.step_count = 0
        self.change_count = 0
        # Restore original parameters
        self._restore_original_params()
        result = self.env.reset(**kwargs)
        return result

    def _restore_original_params(self) -> None:
        """Restore original dynamics parameters."""
        try:
            model = self.unwrapped.model
            if self._original_masses is not None:
                model.body_mass[:] = self._original_masses
            if self._original_frictions is not None:
                model.geom_friction[:] = self._original_frictions
        except AttributeError:
            pass

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Apply dynamics change at regular intervals
        if self.change_interval > 0 and self.step_count % self.change_interval == 0:
            self._apply_dynamics_change()

        # Add change indicator to info
        info["dynamics_changed"] = (
            self.change_interval > 0 and self.step_count % self.change_interval == 0
        )
        info["change_count"] = self.change_count

        return obs, reward, terminated, truncated, info


class SensorDriftWrapper(gym.Wrapper):
    """Wrapper that introduces sensor drift (POMDP scenario).

    Randomly perturbs a subset of observation dimensions to simulate
    sensor aging or malfunction. This tests RIME-PPO's ability to
    use internal state e_t to compensate for partial observability.

    Args:
        env: The environment to wrap.
        drift_rate: Rate at which drift accumulates per step.
        drift_dimensions: Indices of observation dimensions to perturb.
            If None, randomly selects 20% of dimensions.
        reset_probability: Probability of drift resetting each step.
    """

    def __init__(
        self,
        env: gym.Env,
        drift_rate: float = 0.001,
        drift_dimensions: list[int] | None = None,
        reset_probability: float = 0.01,
    ):
        super().__init__(env)
        self.drift_rate = drift_rate
        self.reset_probability = reset_probability

        obs_dim = env.observation_space.shape[0]
        if drift_dimensions is None:
            n_drift = max(1, obs_dim // 5)
            self.drift_dimensions = np.random.choice(obs_dim, n_drift, replace=False).tolist()
        else:
            self.drift_dimensions = drift_dimensions

        self.current_drift = np.zeros(obs_dim)

    def reset(self, **kwargs):
        self.current_drift = np.zeros(self.observation_space.shape[0])
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Accumulate drift
        self.current_drift[self.drift_dimensions] += np.random.normal(
            0, self.drift_rate, len(self.drift_dimensions)
        )

        # Random drift reset (sensor recalibration)
        if np.random.random() < self.reset_probability:
            self.current_drift[self.drift_dimensions] = 0.0

        # Apply drift to observation
        obs = obs + self.current_drift

        return obs, reward, terminated, truncated, info
