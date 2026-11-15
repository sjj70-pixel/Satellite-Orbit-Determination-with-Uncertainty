"""Pure-function checks for the cross-seed orbit determination experiment."""

import numpy as np
import pytest

from experiments.monte_carlo import (
    DURATION_S,
    FIT_END_S,
    noise_realization,
    score_forecast,
    select_psd,
    summarize_validation,
    validate_design,
)
from src.ground_stations import GroundStation
from src.measurements import TrackingMeasurement


def test_noise_is_reproducible_and_does_not_mutate_ideal():
    station = GroundStation("Test", latitude_deg=0.0, longitude_deg=0.0)
    ideal = [
        TrackingMeasurement(
            time_s=123.0,
            station=station,
            range_m=1e6,
            range_rate_mps=123.0,
            range_std_m=0.0,
            range_rate_std_mps=0.0,
        )
    ]
    first = noise_realization(ideal, seed=57)
    second = noise_realization(ideal, seed=57)
    third = noise_realization(ideal, seed=58)
    assert first[0] == second[0]
    assert first[0] != third[0]
    assert first[0].station is station
    assert first[0].range_std_m == 10.0
    assert first[0].range_rate_std_mps == 0.1
    assert ideal[0].range_std_m == 0.0
    assert ideal[0].range_m == 1e6


@pytest.mark.parametrize(
    "calibration,validation,psd_grid",
    [
        ([1], [1], [1e-10]),
        ([1, 1], [2], [1e-10]),
        ([1], [2, 2], [1e-10]),
        ([1], [2], [1e-10, 1e-10]),
        ([], [2], [1e-10]),
        ([1], [2], [-1e-10]),
    ],
)
def test_invalid_calibration_designs_rejected(calibration, validation, psd_grid):
    with pytest.raises(ValueError):
        validate_design(calibration, validation, psd_grid)


def test_forecast_rmse_and_nees():
    times = np.array([FIT_END_S, FIT_END_S + 300.0, DURATION_S])
    truth = np.zeros((3, 6))
    estimated = np.zeros_like(truth)
    estimated[0, :3] = [999.0, 0.0, 0.0]  # Excluded (fit boundary)
    estimated[1:, :3] = [3.0, 4.0, 0.0]
    estimated[1:, 3:] = [0.0, 0.0, 2.0]
    covariance = np.broadcast_to(np.eye(6), (3, 6, 6)).copy()
    scores = score_forecast(times, estimated, truth, covariance)
    assert scores["forecast_position_rmse_m"] == pytest.approx(5.0)
    assert scores["forecast_velocity_rmse_mps"] == pytest.approx(2.0)
    assert scores["position_error_24h_m"] == pytest.approx(5.0)
    assert scores["position_nees_24h"] == pytest.approx(25.0)
    assert scores["predicted_position_scale_24h_m"] == pytest.approx(np.sqrt(3))


def test_select_psd_uses_mean_and_prefers_lower_psd_on_tie():
    rows = [
        {"acceleration_psd": 1e-10, "forecast_position_rmse_m": 2.0},
        {"acceleration_psd": 1e-10, "forecast_position_rmse_m": 4.0},
        {"acceleration_psd": 1e-9, "forecast_position_rmse_m": 3.0},
        {"acceleration_psd": 1e-9, "forecast_position_rmse_m": 3.0},
    ]
    assert select_psd(rows) == 1e-10


def test_summary_contains_only_correct_scenario_and_estimator():
    rows = [
        {
            "truth_model": "Gravity + J2",
            "estimator": "Batch least squares",
            "forecast_position_rmse_m": 4.0,
            "forecast_velocity_rmse_mps": 0.4,
            "position_nees_24h": None,
        },
        {
            "truth_model": "Gravity + J2",
            "estimator": "Batch least squares",
            "forecast_position_rmse_m": 6.0,
            "forecast_velocity_rmse_mps": 0.6,
            "position_nees_24h": None,
        },
        {
            "truth_model": "Gravity + J2 + drag",
            "estimator": "EKF tuned",
            "forecast_position_rmse_m": 10.0,
            "forecast_velocity_rmse_mps": 1.0,
            "position_nees_24h": 5.0,
        },
    ]
    result = summarize_validation(rows)
    nominal_batch = next(r for r in result if r["truth_model"] == "Gravity + J2")
    assert nominal_batch["position_rmse_mean_m"] == pytest.approx(5.0)
    assert nominal_batch["position_rmse_std_m"] == pytest.approx(np.sqrt(2))
    drag_ekf = next(r for r in result if r["truth_model"] == "Gravity + J2 + drag")
    assert drag_ekf["position_nees_24h_mean"] == pytest.approx(5.0)
