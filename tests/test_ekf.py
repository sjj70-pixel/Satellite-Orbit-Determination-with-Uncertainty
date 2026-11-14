"""Tests for the continuous-discrete satellite Extended Kalman Filter."""

import numpy as np
import pytest

from src.constants import MU_EARTH, R_EARTH
from src.dynamics import OrbitalDynamics
from src.ekf import (
    dynamics_jacobian,
    measurement_jacobian,
    predict_step,
    run_ekf,
    update_step,
)
from src.ground_stations import GroundStation
from src.measurements import generate_measurement, predict_measurement
from src.propagation import propagate_orbit
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


def prior_covariance():
    return np.diag([2000.0**2] * 3 + [1.0**2] * 3)


def test_measurement_jacobian_matches_finite_differences():
    station = GroundStation("Example", latitude_deg=30.0, longitude_deg=20.0)
    state = np.array([7.0e6, 3.0e5, 1.0e6, -500.0, 7200.0, 1300.0])
    time_s = 235.0

    analytic = measurement_jacobian(state, station, time_s)
    numeric = np.empty((2, 6))
    for j in range(6):
        step = 0.5 if j < 3 else 0.002
        change = np.zeros(6)
        change[j] = step
        numeric[:, j] = (
            predict_measurement(state + change, station, time_s)
            - predict_measurement(state - change, station, time_s)
        ) / (2.0 * step)

    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-6)


def test_dynamics_jacobian_matches_two_body_gravity_gradient():
    state = np.array([6.8e6, -5e5, 1e6, 0.0, 7000.0, 1000.0])
    jacobian = dynamics_jacobian(state, OrbitalDynamics(include_j2=False))
    radius = np.linalg.norm(state[:3])
    expected_gravity_gradient = MU_EARTH * (
        3 * np.outer(state[:3], state[:3]) / radius**5 - np.eye(3) / radius**3
    )

    np.testing.assert_allclose(jacobian[:3, 3:], np.eye(3), atol=0)
    np.testing.assert_allclose(
        jacobian[3:, :3], expected_gravity_gradient, rtol=1e-8, atol=1e-13
    )
    np.testing.assert_array_equal(jacobian[:3, :3], np.zeros((3, 3)))
    np.testing.assert_array_equal(jacobian[3:, 3:], np.zeros((3, 3)))


def test_single_measurement_correction_reduces_error_and_uncertainty():
    truth = initial_orbit()
    station = GroundStation("A", 0.0, 0.0)
    measurement = generate_measurement(
        truth, station, t=0.0, range_std_m=0.0, range_rate_std_mps=0.0
    )
    assert measurement is not None

    guess = truth + np.array([800.0, 0.0, 0.0, 0.1, 0.0, 0.0])
    prior = prior_covariance()
    prior_range_error = np.abs(
        measurement.vector()[0] - predict_measurement(guess, station, 0.0)[0]
    )

    new_state, posterior, innovation, s, nis = update_step(guess, prior, measurement)
    posterior_range_error = np.abs(
        measurement.vector()[0] - predict_measurement(new_state, station, 0.0)[0]
    )

    assert posterior_range_error < prior_range_error
    assert np.trace(posterior[:3, :3]) < np.trace(prior[:3, :3])
    assert np.min(np.linalg.eigvalsh(posterior)) > -1e-8
    assert innovation.shape == (2,)
    assert s.shape == (2, 2)
    assert nis >= 0.0


def test_simultaneous_stations_are_processed_sequentially():
    truth = initial_orbit()
    station_a = GroundStation("A", 0.0, 0.0)
    station_b = GroundStation("B", 0.0, 10.0, min_elevation_deg=0.0)
    a = generate_measurement(
        truth,
        station_a,
        t=0.0,
        range_std_m=10.0,
        range_rate_std_mps=0.1,
        rng=np.random.default_rng(1),
    )
    b = generate_measurement(
        truth,
        station_b,
        t=0.0,
        range_std_m=10.0,
        range_rate_std_mps=0.1,
        rng=np.random.default_rng(2),
    )
    assert a is not None and b is not None

    guess = truth + np.array([1000.0, -500.0, 200.0, 0.2, -0.1, 0.05])
    prior = prior_covariance()
    first_state, first_covariance, *_ = update_step(guess, prior, b)
    second_state, second_covariance, *_ = update_step(first_state, first_covariance, a)
    result = run_ekf([b, a], guess, prior)

    assert result.update_counts.tolist() == [2]
    np.testing.assert_allclose(result.states[0], second_state)
    np.testing.assert_allclose(result.covariances[0], second_covariance)
    assert result.innovations.shape == (2, 2)
    assert result.nis.shape == (2,)


def test_covariance_propagates_during_a_gap_without_measurements():
    initial = initial_orbit()
    prior = prior_covariance()
    dynamics = OrbitalDynamics(include_j2=True)
    no_noise = run_ekf(
        [],
        initial,
        prior,
        output_times_s=[0.0, 600.0],
        dynamics=dynamics,
        acceleration_psd=0.0,
    )
    with_noise = run_ekf(
        [],
        initial,
        prior,
        output_times_s=[0.0, 600.0],
        dynamics=dynamics,
        acceleration_psd=1e-8,
    )
    expected = propagate_orbit(
        initial, (0.0, 600.0), dynamics=dynamics, t_eval=[600.0]
    ).y[:, -1]

    np.testing.assert_allclose(no_noise.states[-1], expected, rtol=0, atol=0.02)
    np.testing.assert_array_equal(no_noise.update_counts, [0, 0])
    assert np.trace(no_noise.covariances[-1][:3, :3]) > np.trace(prior[:3, :3])
    assert np.trace(with_noise.covariances[-1]) > np.trace(no_noise.covariances[-1])
    np.testing.assert_allclose(
        with_noise.covariances[-1], with_noise.covariances[-1].T, atol=1e-8
    )
    assert np.min(np.linalg.eigvalsh(with_noise.covariances[-1])) > -1e-6


def test_short_tracking_run_improves_state_estimate():
    truth = initial_orbit()
    stations = [
        GroundStation("A", 0.0, 0.0),
        GroundStation("B", 45.0, 80.0),
        GroundStation("C", -35.0, -120.0),
    ]
    simulation = simulate_tracking(
        truth,
        stations,
        duration_s=6 * 3600.0,
        measurement_interval_s=60.0,
        range_std_m=10.0,
        range_rate_std_mps=0.1,
        seed=42,
    )
    guess = truth + np.array([1000.0, -800.0, 500.0, 0.3, -0.2, 0.1])
    result = run_ekf(
        simulation.measurements,
        guess,
        prior_covariance(),
        output_times_s=np.arange(0.0, 6 * 3600.0 + 1, 600.0),
        acceleration_psd=1e-10,
    )

    assert len(result.innovations) == len(simulation.measurements)
    assert int(result.update_counts.sum()) == len(simulation.measurements)
    assert np.all(np.isfinite(result.states))
    assert np.all(np.isfinite(result.covariances))

    # At the initial epoch this run should improve the noisy starting guess
    # once the first visible ground station supplies a measurement.
    first_error = np.linalg.norm(result.states[0][:3] - truth[:3])
    assert first_error < np.linalg.norm(guess[:3] - truth[:3])
    final_position_error = np.linalg.norm(
        result.states[-1][:3] - simulation.true_states[-1, :3]
    )
    assert final_position_error < 100.0


def test_invalid_covariance_and_process_noise_are_rejected():
    initial = initial_orbit()
    with pytest.raises(ValueError):
        run_ekf([], initial, np.eye(5))
    with pytest.raises(ValueError):
        run_ekf([], initial, -np.eye(6))
    with pytest.raises(ValueError):
        run_ekf([], initial, np.eye(6), acceleration_psd=-1.0)
    with pytest.raises(ValueError):
        predict_step(initial, np.eye(6), 20.0, 10.0, OrbitalDynamics(include_j2=True))
