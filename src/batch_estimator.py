"""Batch orbit determination from ground-station range and range-rate data.

Only the six components of the initial Cartesian state are estimated.
The gravitational model is supplied and its physical parameters are fixed.
All positions are meters, velocities are m/s, and times are seconds.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .dynamics import OrbitalDynamics
from .measurements import predict_measurement
from .propagation import propagate_orbit


@dataclass
class BatchEstimate:
    """Initial-state estimate and diagnostics from nonlinear least squares."""

    initial_state: np.ndarray
    success: bool
    message: str
    nfev: int
    cost: float
    weighted_residuals: np.ndarray


def estimate_initial_state(
    measurements,
    initial_guess,
    dynamics=None,
    max_nfev=100,
    position_scale_m=1000.0,
    velocity_scale_mps=1.0,
):
    """Fit a six-dimensional initial state to tracking observations.

    Parameters
    ----------
    measurements : iterable of TrackingMeasurement
        Observations from one satellite, with times since the same t=0 epoch.
        Each observation contains the station geometry and noise levels.
    initial_guess : array_like, shape (6,)
        Starting guess [x,y,z,vx,vy,vz] in ECI coordinates.
    dynamics : OrbitalDynamics, optional
        Model assumed by the estimator; defaults to two-body plus J2.
    max_nfev : int
        Maximum number of objective evaluations by SciPy's optimizer.
    position_scale_m : float
        Characteristic position correction used to scale optimization variables.
    velocity_scale_mps : float
        Characteristic velocity correction used to scale optimization variables.

    Returns
    -------
    BatchEstimate
        Fitted initial state, convergence information, and normalized residuals.

    Notes
    -----
    Residuals are divided by each measurement's noise standard deviation.
    For noiseless synthetic data (reported sigma=0), a numerical floor of
    1 m for range and 0.01 m/s for range rate avoids division by zero.
    The estimator does not have access to the simulation's true state.
    """
    observations = list(measurements)
    if len(observations) < 3:
        raise ValueError("At least three range/range-rate observations are required")

    guess = np.asarray(initial_guess, dtype=float)
    if guess.shape != (6,) or not np.all(np.isfinite(guess)):
        raise ValueError("initial_guess must be a finite array of shape (6,)")

    if not np.isfinite(position_scale_m) or position_scale_m <= 0:
        raise ValueError("position_scale_m must be positive and finite")
    if not np.isfinite(velocity_scale_mps) or velocity_scale_mps <= 0:
        raise ValueError("velocity_scale_mps must be positive and finite")

    if dynamics is None:
        dynamics = OrbitalDynamics(include_j2=True)

    observation_times = np.array([m.time_s for m in observations], dtype=float)
    if not np.all(np.isfinite(observation_times)) or np.any(observation_times < 0):
        raise ValueError("All measurement times must be finite and nonnegative")

    measured_values = np.array([m.vector() for m in observations], dtype=float)
    reported_std = np.array(
        [[m.range_std_m, m.range_rate_std_mps] for m in observations], dtype=float
    )
    if not np.all(np.isfinite(measured_values)):
        raise ValueError("Measured values must be finite")
    if not np.all(np.isfinite(reported_std)) or np.any(reported_std < 0):
        raise ValueError(
            "Measurement standard deviations must be nonnegative and finite"
        )

    # Use measured uncertainties when positive; floors handle noiseless tests.
    standard_deviations = np.maximum(reported_std, [1.0, 0.01])
    unique_times, time_indices = np.unique(observation_times, return_inverse=True)
    t_end = float(unique_times[-1])

    scales = np.array(
        [
            position_scale_m,
            position_scale_m,
            position_scale_m,
            velocity_scale_mps,
            velocity_scale_mps,
            velocity_scale_mps,
        ]
    )

    def residuals(scaled_correction):
        trial_state = guess + scales * scaled_correction

        # Propagate once for all unique observation times, rather than
        # independently integrating for every ground-station measurement.
        if t_end == 0.0:
            states_at_times = trial_state[np.newaxis, :]
        else:
            solution = propagate_orbit(
                initial_state=trial_state,
                t_span=(0.0, t_end),
                dynamics=dynamics,
                t_eval=unique_times,
            )
            states_at_times = solution.y.T

        predictions = np.array(
            [
                predict_measurement(
                    states_at_times[time_indices[k]],
                    observation.station,
                    observation.time_s,
                )
                for k, observation in enumerate(observations)
            ]
        )
        return ((predictions - measured_values) / standard_deviations).ravel()

    # Central differences in scaled coordinates correspond to perturbing
    # position by 10 m and velocity by 0.001 m/s at the default scales.
    # These finite steps reduce sensitivity to numerical integration noise.
    physical_jac_steps = np.array([10.0] * 3 + [0.001] * 3)
    scaled_jac_steps = physical_jac_steps / scales

    def jacobian(scaled_correction: np.ndarray) -> np.ndarray:
        matrix = np.empty((2 * len(observations), 6), dtype=float)
        for j, step in enumerate(scaled_jac_steps):
            change = np.zeros(6)
            change[j] = step
            matrix[:, j] = (
                residuals(scaled_correction + change)
                - residuals(scaled_correction - change)
            ) / (2.0 * step)
        return matrix

    result = least_squares(
        residuals,
        x0=np.zeros(6),
        jac=jacobian,  # type: ignore[reportArgumentType]
        method="trf",
        max_nfev=max_nfev,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )

    return BatchEstimate(
        initial_state=guess + scales * result.x,
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        cost=float(result.cost),
        weighted_residuals=result.fun.copy(),
    )
