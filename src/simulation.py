"""Generate synthetic satellite tracking data from orbital dynamics.

The true trajectory is kept separate from the noisy observations so that
orbit-estimation algorithms can receive measurements without ground truth.

All positions/ranges are meters, velocities/range rates are m/s, and times
are seconds since the simulation epoch. GroundStation uses the same epoch.
"""

from dataclasses import dataclass

import numpy as np

from .dynamics import OrbitalDynamics
from .measurements import TrackingMeasurement, generate_measurement
from .propagation import propagate_orbit


@dataclass
class TrackingSimulation:
    """Result of a synthetic tracking simulation.

    Attributes
    ----------
    times_s : ndarray, shape (N,)
        Times at which ground truth was sampled.
    true_states : ndarray, shape (N, 6)
        Satellite position and velocity at each sampled time.
    measurements : list[TrackingMeasurement]
        Noisy observations, ordered by time and then station.
    """

    times_s: np.ndarray
    true_states: np.ndarray
    measurements: list[TrackingMeasurement]

    def measurement_rows(self):
        """Return observations as records suitable for a pandas DataFrame.

        The records deliberately exclude the true satellite state.
        """
        return [
            {
                "time_s": m.time_s,
                "station": m.station.name,
                "range_m": m.range_m,
                "range_rate_mps": m.range_rate_mps,
                "range_std_m": m.range_std_m,
                "range_rate_std_mps": m.range_rate_std_mps,
            }
            for m in self.measurements
        ]


def simulate_tracking(
    initial_state,
    stations,
    duration_s,
    measurement_interval_s=30.0,
    dynamics=None,
    range_std_m=10.0,
    range_rate_std_mps=0.1,
    seed=42,
):
    """Propagate one satellite and simulate ground-station observations.

    Parameters
    ----------
    initial_state : array_like, shape (6,)
        Initial satellite position (m) and velocity (m/s) in ECI.
    stations : iterable of GroundStation
        Ground stations that may observe the satellite.
    duration_s : float
        Simulation duration, starting at t=0 (seconds).
    measurement_interval_s : float
        Time between attempted observations for each station (seconds).
    dynamics : OrbitalDynamics, optional
        Dynamics used to generate ground truth. Defaults to J2 gravity.
    range_std_m : float
        Standard deviation of range noise (m).
    range_rate_std_mps : float
        Standard deviation of range-rate noise (m/s).
    seed : int or None
        Seed for reproducible measurement noise.

    Returns
    -------
    TrackingSimulation
        Includes both hidden ground truth and noisy observations. Pass only
        its measurements to an estimator, never its true_states.

    Notes
    -----
    Attempts occur at t=0, dt, 2*dt, ... up to duration_s. Each station
    provides an observation only when the satellite meets its elevation mask.
    No observations are interpolated or generated between these times.
    """
    state = np.asarray(initial_state, dtype=float)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must be a finite array of shape (6,)")

    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive and finite")
    if not np.isfinite(measurement_interval_s) or measurement_interval_s <= 0:
        raise ValueError("measurement_interval_s must be positive and finite")
    if not np.isfinite(range_std_m) or range_std_m < 0:
        raise ValueError("range_std_m must be finite and nonnegative")
    if not np.isfinite(range_rate_std_mps) or range_rate_std_mps < 0:
        raise ValueError("range_rate_std_mps must be finite and nonnegative")

    stations = tuple(stations)
    if not stations:
        raise ValueError("At least one ground station is required")

    if dynamics is None:
        dynamics = OrbitalDynamics(include_j2=True)

    # Sample only at scheduled measurement times, all within t_span.
    num_steps = int(np.floor(duration_s / measurement_interval_s))
    times = measurement_interval_s * np.arange(num_steps + 1, dtype=float)
    times = times[times <= duration_s]

    trajectory = propagate_orbit(
        initial_state=state,
        t_span=(0.0, duration_s),
        dynamics=dynamics,
        t_eval=times,
    )
    true_states = trajectory.y.T.copy()  # Shape (N, 6)

    rng = np.random.default_rng(seed)
    measurements = []

    # Loop over times first so observations are chronological.
    for t, satellite_state in zip(times, true_states):
        for station in stations:
            measurement = generate_measurement(
                satellite_state=satellite_state,
                station=station,
                t=t,
                range_std_m=range_std_m,
                range_rate_std_mps=range_rate_std_mps,
                rng=rng,
            )
            if measurement is not None:
                measurements.append(measurement)

    return TrackingSimulation(
        times_s=times,
        true_states=true_states,
        measurements=measurements,
    )
