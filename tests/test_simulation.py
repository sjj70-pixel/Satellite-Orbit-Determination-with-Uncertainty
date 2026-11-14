"""Integration tests for synthetic satellite tracking simulations."""

import numpy as np
import pytest

from src.constants import MU_EARTH, R_EARTH
from src.dynamics import OrbitalDynamics
from src.ground_stations import GroundStation
from src.measurements import predict_measurement
from src.simulation import simulate_tracking


def circular_initial_state():
    radius = R_EARTH + 500_000.0
    speed = np.sqrt(MU_EARTH / radius)
    return np.array([radius, 0.0, 0.0, 0.0, speed, 0.0])


def test_sampling_and_true_state_shape():
    state = circular_initial_state()
    station = GroundStation("Equator", 0.0, 0.0)
    result = simulate_tracking(
        state,
        stations=[station],
        duration_s=600.0,
        measurement_interval_s=30.0,
    )

    np.testing.assert_allclose(result.times_s, np.arange(21) * 30.0)
    assert result.true_states.shape == (21, 6)
    np.testing.assert_allclose(result.true_states[0], state)


def test_visibility_and_multiple_stations():
    state = circular_initial_state()
    stations = [
        GroundStation("Station A", 0.0, 0.0),
        GroundStation("Station B", 0.0, 180.0),
    ]
    result = simulate_tracking(
        state,
        stations=stations,
        duration_s=6000.0,
        measurement_interval_s=30.0,
    )

    observed_stations = {m.station.name for m in result.measurements}
    assert observed_stations == {"Station A", "Station B"}
    assert len(result.measurements) < len(result.times_s) * len(stations)

    for measurement in result.measurements:
        index = int(round(measurement.time_s / 30.0))
        satellite_position = result.true_states[index, :3]
        assert measurement.station.is_visible(satellite_position, measurement.time_s)


def test_zero_noise_observations_match_prediction():
    state = circular_initial_state()
    station = GroundStation("Equator", 0.0, 0.0)
    result = simulate_tracking(
        state,
        stations=[station],
        duration_s=600.0,
        measurement_interval_s=30.0,
        range_std_m=0.0,
        range_rate_std_mps=0.0,
    )

    assert result.measurements
    for measurement in result.measurements:
        index = int(round(measurement.time_s / 30.0))
        expected = predict_measurement(
            result.true_states[index], station, measurement.time_s
        )
        np.testing.assert_allclose(measurement.vector(), expected)


def test_seeded_noise_is_reproducible():
    state = circular_initial_state()
    station = GroundStation("Equator", 0.0, 0.0)
    kwargs = {
        "stations": [station],
        "duration_s": 600.0,
        "dynamics": OrbitalDynamics(include_j2=False),
        "seed": 17,
    }
    first = simulate_tracking(state, **kwargs)
    second = simulate_tracking(state, **kwargs)

    assert first.measurement_rows() == second.measurement_rows()
    np.testing.assert_array_equal(first.true_states, second.true_states)
    assert "true_state" not in first.measurement_rows()[0]


def test_rejects_invalid_simulation_settings():
    state = circular_initial_state()
    station = GroundStation("Equator", 0.0, 0.0)

    with pytest.raises(ValueError):
        simulate_tracking(state, [], duration_s=600.0)
    with pytest.raises(ValueError):
        simulate_tracking(state, [station], duration_s=-1.0)
    with pytest.raises(ValueError):
        simulate_tracking(
            state,
            [station],
            duration_s=600.0,
            measurement_interval_s=0.0,
        )
