"""Continuous-discrete Extended Kalman Filter for satellite orbit determination.

State: [x, y, z, vx, vy, vz] in ECI (meters, meters/second).
Measurements: ground-station range (m) and range rate (m/s).

Between observations, propagate the nonlinear state and its covariance:
    x_dot = f(x)
    P_dot = A P + P A.T + Qc,  A = df/dx.

At each observation, use the nonlinear measurement model h(x) and its
Jacobian H = dh/dx. Joseph-form updates help preserve covariance symmetry.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import OrbitalDynamics
from .measurements import predict_measurement


@dataclass
class EKFResult:
    """Filtered states and covariances, including times without observations.

    Each row of states and covariances represents the estimate AFTER all
    available measurements at that time have been processed.
    Innovations and their squared Mahalanobis distances (NIS) are stored in
    chronological measurement order, including multiple stations at one time.
    """

    times_s: np.ndarray  # Shape (N,)
    states: np.ndarray  # Shape (N, 6)
    covariances: np.ndarray  # Shape (N, 6, 6)
    update_counts: np.ndarray  # Shape (N,) ; 0 when no station reports
    innovations: np.ndarray  # Shape (M, 2)
    innovation_covariances: np.ndarray  # Shape (M, 2, 2)
    nis: np.ndarray  # Shape (M,)


def dynamics_jacobian(state, dynamics, position_step_m=10.0):
    """Linearize two-body/J2 orbital dynamics around a 6D Cartesian state.

    Since these gravity accelerations depend on position but not velocity,
    A has blocks [[0, I], [d(acceleration)/d(position), 0]].
    The lower-left block uses central finite differences.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (6,):
        raise ValueError("state must have shape (6,)")
    if not np.isfinite(position_step_m) or position_step_m <= 0:
        raise ValueError("position_step_m must be positive and finite")

    a = np.zeros((6, 6), dtype=float)
    a[:3, 3:] = np.eye(3)
    for j in range(3):
        delta = np.zeros(3)
        delta[j] = position_step_m
        a[3:, j] = (
            dynamics.acceleration(state[:3] + delta)
            - dynamics.acceleration(state[:3] - delta)
        ) / (2.0 * position_step_m)
    return a


def measurement_jacobian(state, station, time_s):
    """Return the analytic 2x6 range/range-rate Jacobian H.

    Let d = r_sat - r_station, v = v_sat - v_station, u = d / |d|.
    Then range = |d|, range_rate = u dot v, and
        d(range_rate)/d(r_sat) = (v - range_rate*u) / |d|.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("state must be a finite array of shape (6,)")

    relative_position = state[:3] - station.position_eci(time_s)
    relative_velocity = state[3:] - station.velocity_eci(time_s)
    distance = np.linalg.norm(relative_position)
    if distance == 0.0:
        raise ValueError("satellite and station cannot have identical positions")

    unit_line_of_sight = relative_position / distance
    range_rate = np.dot(unit_line_of_sight, relative_velocity)

    h = np.zeros((2, 6), dtype=float)
    h[0, :3] = unit_line_of_sight
    h[1, :3] = (relative_velocity - range_rate * unit_line_of_sight) / distance
    h[1, 3:] = unit_line_of_sight
    return h


def predict_step(state, covariance, start_s, end_s, dynamics, acceleration_psd=1e-10):
    """Propagate the state and 6x6 covariance to a later time.

    acceleration_psd is the white-acceleration noise spectral density,
    in m^2/s^3. Set it to zero for a perfectly modeled gravity baseline.
    """
    state = np.asarray(state, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if end_s < start_s:
        raise ValueError("end_s must not precede start_s")
    if not np.isfinite(acceleration_psd) or acceleration_psd < 0:
        raise ValueError("acceleration_psd must be finite and nonnegative")
    if end_s == start_s:
        return state.copy(), covariance.copy()

    qc = np.zeros((6, 6))
    qc[3:, 3:] = acceleration_psd * np.eye(3)

    def state_and_covariance_rhs(time_s, combined):
        x = combined[:6]
        p = combined[6:].reshape(6, 6)
        a = dynamics_jacobian(x, dynamics)
        dp = a @ p + p @ a.T + qc
        return np.concatenate((dynamics(time_s, x), dp.ravel()))

    initial = np.concatenate((state, covariance.ravel()))
    solution = solve_ivp(
        state_and_covariance_rhs,
        (float(start_s), float(end_s)),
        initial,
        method="DOP853",
        rtol=1e-9,
        atol=np.concatenate((np.full(3, 1e-3), np.full(3, 1e-6), np.full(36, 1e-8))),
    )
    if not solution.success:
        raise RuntimeError(f"EKF prediction failed: {solution.message}")

    final = solution.y[:, -1]
    p = final[6:].reshape(6, 6)
    return final[:6], 0.5 * (p + p.T)


def update_step(state, covariance, measurement):
    """Apply one ground-station observation; return state, P, innovation, S, NIS.

    R is diagonal because the measurement generator assumes independent range
    and range-rate noise. Zero reported sigmas use the same numerical floors
    as our batch estimator (1 m and 0.01 m/s).
    """
    observed = measurement.vector()
    sigmas = np.array([measurement.range_std_m, measurement.range_rate_std_mps])
    if not np.all(np.isfinite(observed)):
        raise ValueError("Measurement values must be finite")
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas < 0):
        raise ValueError(
            "Measurement standard deviations must be nonnegative and finite"
        )

    r = np.diag(np.maximum(sigmas, [1.0, 0.01]) ** 2)
    h = measurement_jacobian(state, measurement.station, measurement.time_s)
    prediction = predict_measurement(state, measurement.station, measurement.time_s)
    innovation = observed - prediction
    s = h @ covariance @ h.T + r

    # Avoid forming S^{-1} explicitly. K = P H.T S^{-1}.
    k = np.linalg.solve(s, h @ covariance).T
    new_state = state + k @ innovation
    identity_minus_kh = np.eye(6) - k @ h
    new_covariance = identity_minus_kh @ covariance @ identity_minus_kh.T + k @ r @ k.T
    new_covariance = 0.5 * (new_covariance + new_covariance.T)
    nis = float(innovation @ np.linalg.solve(s, innovation))
    return new_state, new_covariance, innovation, s, nis


def run_ekf(
    measurements,
    initial_state,
    initial_covariance,
    output_times_s=None,
    dynamics=None,
    acceleration_psd=1e-10,
):
    """Filter tracking observations in chronological order from epoch t=0.

    Parameters
    ----------
    measurements : iterable of TrackingMeasurement
        May contain simultaneous observations from different stations.
        Empty lists are valid: the filter then only predicts.
    initial_state : array_like, shape (6,)
        Initial position/velocity estimate at t=0 (NOT the true state).
    initial_covariance : array_like, shape (6, 6)
        Prior error covariance at t=0, in squared state units.
    output_times_s : array_like, optional
        Additional times at which to record estimates, including times without
        any observations. Measurement times and t=0 are always included.
    dynamics : OrbitalDynamics, optional
        Two-body + J2 by default; assumed gravity model for the filter.
    acceleration_psd : float
        White-acceleration process-noise spectral density (m^2/s^3).

    Returns
    -------
    EKFResult
        Estimated states, covariance history, and innovation statistics.
    """
    x = np.asarray(initial_state, dtype=float).copy()
    p = np.asarray(initial_covariance, dtype=float).copy()
    if x.shape != (6,) or not np.all(np.isfinite(x)):
        raise ValueError("initial_state must be a finite array of shape (6,)")
    if p.shape != (6, 6) or not np.all(np.isfinite(p)):
        raise ValueError("initial_covariance must be a finite 6x6 matrix")
    if not np.allclose(p, p.T, rtol=0.0, atol=1e-8):
        raise ValueError("initial_covariance must be symmetric")
    p = 0.5 * (p + p.T)
    if np.min(np.linalg.eigvalsh(p)) < -1e-8:
        raise ValueError("initial_covariance must be positive semidefinite")
    if not np.isfinite(acceleration_psd) or acceleration_psd < 0:
        raise ValueError("acceleration_psd must be finite and nonnegative")

    observations = sorted(measurements, key=lambda m: m.time_s)
    observation_times = np.array([m.time_s for m in observations], dtype=float)
    if not np.all(np.isfinite(observation_times)) or np.any(observation_times < 0):
        raise ValueError("Measurement times must be finite and nonnegative")

    if output_times_s is None:
        extra_times = np.array([], dtype=float)
    else:
        extra_times = np.asarray(output_times_s, dtype=float)
        if extra_times.ndim != 1:
            raise ValueError("output_times_s must be a one-dimensional array")
        if not np.all(np.isfinite(extra_times)) or np.any(extra_times < 0):
            raise ValueError("output_times_s must be finite and nonnegative")

    # All station updates at a particular time are applied before recording.
    times = np.unique(np.concatenate(([0.0], extra_times, observation_times)))
    states = np.empty((len(times), 6))
    covariances = np.empty((len(times), 6, 6))
    update_counts = np.zeros(len(times), dtype=int)
    innovations = []
    innovation_covariances = []
    nis_values = []
    if dynamics is None:
        dynamics = OrbitalDynamics(include_j2=True)

    previous_time = 0.0
    measurement_index = 0
    for i, time_s in enumerate(times):
        x, p = predict_step(x, p, previous_time, time_s, dynamics, acceleration_psd)
        previous_time = time_s

        while (
            measurement_index < len(observations)
            and observations[measurement_index].time_s == time_s
        ):
            x, p, innovation, s, nis = update_step(
                x, p, observations[measurement_index]
            )
            innovations.append(innovation)
            innovation_covariances.append(s)
            nis_values.append(nis)
            update_counts[i] += 1
            measurement_index += 1

        states[i] = x
        covariances[i] = p

    return EKFResult(
        times_s=times,
        states=states,
        covariances=covariances,
        update_counts=update_counts,
        innovations=np.asarray(innovations, dtype=float).reshape(-1, 2),
        innovation_covariances=np.asarray(innovation_covariances, dtype=float).reshape(
            -1, 2, 2
        ),
        nis=np.asarray(nis_values, dtype=float),
    )
