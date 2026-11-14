"""Tests for batch least-squares orbit determination."""

import numpy as np
import pytest

from src.batch_estimator import estimate_initial_state
from src.constants import MU_EARTH, R_EARTH
from src.ground_stations import GroundStation
from src.simulation import simulate_tracking


def initial_orbit():
    radius = R_EARTH + 500_000.0
    speed = np.sqrt(MU_EARTH / radius)
    inclination = np.deg2rad(55.0)
    return np.array(
        [
            radius,
            0.0,
            0.0,
            0.0,
            speed * np.cos(inclination),
            speed * np.sin(inclination),
        ]
    )


def three_stations():
    return [
        GroundStation("Station A", latitude_deg=0.0, longitude_deg=0.0),
        GroundStation("Station B", latitude_deg=45.0, longitude_deg=80.0),
        GroundStation("Station C", latitude_deg=-35.0, longitude_deg=-120.0),
    ]


def test_recovers_initial_state_from_noiseless_data():
    truth = initial_orbit()
    simulation = simulate_tracking(
        truth,
        three_stations(),
        duration_s=6 * 3600.0,
        measurement_interval_s=60.0,
        range_std_m=0.0,
        range_rate_std_mps=0.0,
        seed=7,
    )
    guess = truth + np.array([1000.0, -800.0, 500.0, 0.3, -0.2, 0.1])

    estimate = estimate_initial_state(simulation.measurements, guess)

    assert estimate.success, estimate.message
    assert np.linalg.norm(estimate.initial_state[:3] - truth[:3]) < 5.0
    assert np.linalg.norm(estimate.initial_state[3:] - truth[3:]) < 0.005


def test_noisy_data_improves_initial_guess():
    truth = initial_orbit()
    simulation = simulate_tracking(
        truth,
        three_stations(),
        duration_s=6 * 3600.0,
        measurement_interval_s=60.0,
        range_std_m=10.0,
        range_rate_std_mps=0.1,
        seed=42,
    )
    guess = truth + np.array([1000.0, -800.0, 500.0, 0.3, -0.2, 0.1])

    estimate = estimate_initial_state(simulation.measurements, guess)

    assert estimate.success, estimate.message
    assert np.linalg.norm(estimate.initial_state[:3] - truth[:3]) < 300.0
    assert np.linalg.norm(estimate.initial_state[3:] - truth[3:]) < 0.2
    assert estimate.cost > 0.0


def test_rejects_insufficient_observations():
    with pytest.raises(ValueError):
        estimate_initial_state([], initial_orbit())
