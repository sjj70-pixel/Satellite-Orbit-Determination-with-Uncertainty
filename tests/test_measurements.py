"""Tests for satellite range and range-rate measurement simulation."""

import numpy as np
import pytest

from src.constants import OMEGA_EARTH, R_EARTH
from src.ground_stations import GroundStation
from src.measurements import generate_measurement, predict_measurement


def equatorial_station():
    return GroundStation("Equator", latitude_deg=0.0, longitude_deg=0.0)


def test_overhead_range_is_altitude():
    station = equatorial_station()
    state = np.array([R_EARTH + 500_000.0, 0, 0, 0, 0, 0])

    predicted = predict_measurement(state, station, 0.0)
    assert np.isclose(predicted[0], 500_000.0)
    assert np.isclose(predicted[1], 0.0)


@pytest.mark.parametrize("radial_velocity", [-100.0, 100.0])
def test_range_rate_sign(radial_velocity):
    station = equatorial_station()
    position = np.array([R_EARTH + 500_000.0, 0.0, 0.0])
    # Account for station's eastward velocity; relative motion is radial.
    satellite_velocity = station.velocity_eci(0.0) + np.array(
        [radial_velocity, 0.0, 0.0]
    )
    state = np.concatenate([position, satellite_velocity])

    predicted = predict_measurement(state, station, 0.0)
    assert np.isclose(predicted[1], radial_velocity)


def test_range_rate_includes_earth_rotation():
    station = equatorial_station()
    state = np.array(
        [
            R_EARTH + 500_000.0,
            500_000.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    predicted = predict_measurement(state, station, 0.0)
    assert np.isclose(predicted[0], 500_000.0 * np.sqrt(2.0))
    assert np.isclose(predicted[1], -OMEGA_EARTH * R_EARTH / np.sqrt(2.0))


def test_hidden_satellite_produces_no_measurement():
    station = equatorial_station()
    state = np.array([-R_EARTH - 500_000.0, 0, 0, 0, 0, 0])
    assert generate_measurement(state, station, 0.0) is None


def test_zero_noise_gives_ideal_values():
    station = equatorial_station()
    state = np.array([R_EARTH + 500_000.0, 0, 0, 100, 0, 0])

    measured = generate_measurement(
        state,
        station,
        0.0,
        range_std_m=0.0,
        range_rate_std_mps=0.0,
        rng=np.random.default_rng(42),
    )

    assert measured is not None
    np.testing.assert_allclose(measured.vector(), [500_000.0, 100.0])
    np.testing.assert_array_equal(measured.covariance(), np.zeros((2, 2)))


def test_noise_is_reproducible_with_fixed_seed():
    station = equatorial_station()
    state = np.array([R_EARTH + 500_000.0, 0, 0, 100, 0, 0])

    first = generate_measurement(state, station, 0.0, rng=np.random.default_rng(7))
    second = generate_measurement(state, station, 0.0, rng=np.random.default_rng(7))

    assert first is not None and second is not None
    np.testing.assert_array_equal(first.vector(), second.vector())
    np.testing.assert_array_equal(first.covariance(), np.diag([10.0**2, 0.1**2]))


def test_invalid_input_rejected():
    station = equatorial_station()
    with pytest.raises(ValueError):
        predict_measurement(np.zeros(5), station, 0.0)
    with pytest.raises(ValueError):
        generate_measurement(np.zeros(6), station, 0.0, range_std_m=-1.0)
