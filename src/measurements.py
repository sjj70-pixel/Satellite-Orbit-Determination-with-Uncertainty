"""Idealized ground-based satellite range and range-rate measurements.

Satellite states are Earth-centered inertial (ECI) Cartesian vectors:
    [x, y, z, vx, vy, vz]

Positions and ranges are in meters, velocities and range rates in m/s,
and times in seconds from the same simulation epoch as GroundStation.
The measurement model neglects light-time, atmospheric, and clock errors.
"""

from dataclasses import dataclass

import numpy as np

from .ground_stations import GroundStation


@dataclass(frozen=True)
class TrackingMeasurement:
    """One noisy observation from a ground station.

    The standard deviations are stored alongside the observations so that
    future estimators can construct the measurement-noise covariance.
    """

    time_s: float
    station: GroundStation
    range_m: float
    range_rate_mps: float
    range_std_m: float
    range_rate_std_mps: float

    def vector(self):
        """Return [range (m), range rate (m/s)] for an estimator."""
        return np.array([self.range_m, self.range_rate_mps])

    def covariance(self):
        """Return the 2x2 covariance for independent measurement errors."""
        return np.diag(
            [
                self.range_std_m**2,
                self.range_rate_std_mps**2,
            ]
        )


def predict_measurement(satellite_state, station, t):
    """Compute ideal [range, range rate] from a satellite state.

    Parameters
    ----------
    satellite_state : array_like, shape (6,)
        Satellite position (m) and velocity (m/s) in ECI coordinates.
    station : GroundStation
        Ground station observing the satellite.
    t : float
        Time in seconds since the simulation epoch.

    Returns
    -------
    ndarray, shape (2,)
        True range in meters and true range rate in m/s.

    Notes
    -----
    This function only computes geometry. It does not enforce visibility;
    generate_measurement() handles the station's elevation mask.
    """
    state = np.asarray(satellite_state, dtype=float)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("satellite_state must be a finite array of shape (6,)")

    satellite_position = state[:3]
    satellite_velocity = state[3:]

    relative_position = satellite_position - station.position_eci(t)
    relative_velocity = satellite_velocity - station.velocity_eci(t)

    distance = np.linalg.norm(relative_position)
    if distance == 0.0:
        raise ValueError("satellite and ground station cannot occupy the same position")

    range_rate = np.dot(relative_position, relative_velocity) / distance
    return np.array([distance, range_rate])


def generate_measurement(
    satellite_state,
    station,
    t,
    range_std_m=10.0,
    range_rate_std_mps=0.1,
    rng=None,
):
    """Generate one noisy range/range-rate observation if visible.

    Parameters
    ----------
    satellite_state : array_like, shape (6,)
        Satellite state in ECI coordinates.
    station : GroundStation
        Observing ground station.
    t : float
        Simulation time in seconds.
    range_std_m : float
        Standard deviation of range noise in meters.
    range_rate_std_mps : float
        Standard deviation of range-rate noise in m/s.
    rng : numpy.random.Generator, optional
        Pass a seeded generator for reproducible simulated observations.

    Returns
    -------
    TrackingMeasurement or None
        None if the satellite is below the station's elevation mask.
    """
    if not np.isfinite(range_std_m) or range_std_m < 0.0:
        raise ValueError("range_std_m must be finite and nonnegative")
    if not np.isfinite(range_rate_std_mps) or range_rate_std_mps < 0.0:
        raise ValueError("range_rate_std_mps must be finite and nonnegative")

    state = np.asarray(satellite_state, dtype=float)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("satellite_state must be a finite array of shape (6,)")

    if not station.is_visible(state[:3], t):
        return None

    ideal_range, ideal_range_rate = predict_measurement(state, station, t)
    if rng is None:
        rng = np.random.default_rng()

    return TrackingMeasurement(
        time_s=float(t),
        station=station,
        range_m=float(ideal_range + rng.normal(0.0, range_std_m)),
        range_rate_mps=float(ideal_range_rate + rng.normal(0.0, range_rate_std_mps)),
        range_std_m=float(range_std_m),
        range_rate_std_mps=float(range_rate_std_mps),
    )
