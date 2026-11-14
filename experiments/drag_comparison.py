"""Compare batch least squares and EKF with perfect and imperfect dynamics.

Run from the project root:
    python -m experiments.drag_comparison

Two 24-hour truth simulations:
    (1) gravity + J2,
    (2) gravity + J2 + atmospheric drag.

For EACH truth scenario, both estimators receive the same noisy tracking
measurements from hours 0-12 and know only gravity + J2. Hours 12-24 are a
genuine held-out forecast: neither method receives more measurements.

This is a single-seed baseline, not a Monte Carlo performance study.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.batch_estimator import estimate_initial_state
from src.drag import DragAugmentedDynamics
from src.dynamics import OrbitalDynamics
from src.ekf import run_ekf
from src.propagation import propagate_orbit
from src.simulation import simulate_tracking

from .tracking_demo import example_stations, initial_orbit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

DURATION_S = 24.0 * 3600.0
FIT_END_S = 12.0 * 3600.0
MEASUREMENT_INTERVAL_S = 30.0
COMPARISON_INTERVAL_S = 300.0
ACCELERATION_PSD = 1e-10  # EKF process noise spectral density (m^2/s^3)
MEASUREMENT_SEED = 42


@dataclass
class ScenarioResult:
    """Tracking errors and uncertainty for a single truth model."""

    label: str
    times_s: np.ndarray
    truth_states: np.ndarray
    batch_position_error_m: np.ndarray
    batch_velocity_error_mps: np.ndarray
    ekf_position_error_m: np.ndarray
    ekf_velocity_error_mps: np.ndarray
    ekf_position_uncertainty_scale_m: np.ndarray
    n_fit_measurements: int
    n_heldout_measurements: int


def rms(values):
    """Root-mean-square of a one-dimensional series."""
    return float(np.sqrt(np.mean(np.square(values))))


def compare_one_truth_model(label, truth_dynamics, stations):
    """Simulate one scenario, then fit both estimators to its same observations."""
    true_initial_state = initial_orbit()
    estimator_dynamics = OrbitalDynamics(include_j2=True)
    simulation = simulate_tracking(
        initial_state=true_initial_state,
        stations=stations,
        duration_s=DURATION_S,
        measurement_interval_s=MEASUREMENT_INTERVAL_S,
        dynamics=truth_dynamics,
        range_std_m=10.0,
        range_rate_std_mps=0.1,
        seed=MEASUREMENT_SEED,
    )

    fitting_measurements = [m for m in simulation.measurements if m.time_s <= FIT_END_S]
    heldout_measurements = [m for m in simulation.measurements if m.time_s > FIT_END_S]
    if len(fitting_measurements) < 3 or not heldout_measurements:
        raise RuntimeError(f"Insufficient ground-station coverage for {label}")

    # The estimators share exactly the same imperfect initial state guess.
    initial_guess = true_initial_state + np.array(
        [1000.0, -800.0, 500.0, 0.3, -0.2, 0.1]
    )
    batch = estimate_initial_state(
        measurements=fitting_measurements,
        initial_guess=initial_guess,
        dynamics=estimator_dynamics,  # No drag, even for drag-affected truth
    )
    if not batch.success:
        raise RuntimeError(f"Batch optimization failed for {label}: {batch.message}")

    sample_times = np.arange(
        0.0, DURATION_S + COMPARISON_INTERVAL_S, COMPARISON_INTERVAL_S
    )
    ekf = run_ekf(
        measurements=fitting_measurements,
        initial_state=initial_guess,
        initial_covariance=np.diag([2000.0**2] * 3 + [1.0**2] * 3),
        output_times_s=sample_times,
        dynamics=estimator_dynamics,  # No drag, even for drag-affected truth
        acceleration_psd=ACCELERATION_PSD,
    )

    # Compare both estimators at precisely the same timestamps.
    ekf_indices = np.searchsorted(ekf.times_s, sample_times)
    truth_indices = np.searchsorted(simulation.times_s, sample_times)
    if not np.array_equal(ekf.times_s[ekf_indices], sample_times) or not np.array_equal(
        simulation.times_s[truth_indices], sample_times
    ):
        raise RuntimeError("Comparison times must lie on both time grids")

    truth_states = simulation.true_states[truth_indices]
    ekf_states = ekf.states[ekf_indices]
    ekf_covariances = ekf.covariances[ekf_indices]
    batch_states = propagate_orbit(
        initial_state=batch.initial_state,
        t_span=(0.0, DURATION_S),
        t_eval=sample_times,
        dynamics=estimator_dynamics,
    ).y.T

    batch_errors = batch_states - truth_states
    ekf_errors = ekf_states - truth_states
    ekf_position_covariances = ekf_covariances[:, :3, :3]

    return ScenarioResult(
        label=label,
        times_s=sample_times,
        truth_states=truth_states,
        batch_position_error_m=np.linalg.norm(batch_errors[:, :3], axis=1),
        batch_velocity_error_mps=np.linalg.norm(batch_errors[:, 3:], axis=1),
        ekf_position_error_m=np.linalg.norm(ekf_errors[:, :3], axis=1),
        ekf_velocity_error_mps=np.linalg.norm(ekf_errors[:, 3:], axis=1),
        ekf_position_uncertainty_scale_m=np.sqrt(
            np.maximum(np.trace(ekf_position_covariances, axis1=1, axis2=2), 0.0)
        ),
        n_fit_measurements=len(fitting_measurements),
        n_heldout_measurements=len(heldout_measurements),
    )


def decorate_plot(ax, label, forecast_shading=True):
    """Mark the boundary between estimation and held-out forecasting."""
    ax.axvline(FIT_END_S / 3600.0, color="black", linestyle="--", lw=1)
    if forecast_shading:
        ax.axvspan(FIT_END_S / 3600.0, DURATION_S / 3600.0, color="gray", alpha=0.1)
    ax.set_title(label)
    ax.set_xlim(0.0, 24.0)
    ax.grid(True, which="both", alpha=0.23)


def plot_errors(cases, quantity, path):
    """Plot batch and EKF error magnitudes over the full 24 hours."""
    if quantity == "position":
        fields = ("batch_position_error_m", "ekf_position_error_m")
        ylabel, title = "Position error (m)", "Position"
    else:
        fields = ("batch_velocity_error_mps", "ekf_velocity_error_mps")
        ylabel, title = "Velocity error (m/s)", "Velocity"

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7.6), sharex=True, constrained_layout=True
    )
    for ax, case in zip(axes, cases):
        hours = case.times_s / 3600.0
        ax.semilogy(hours, getattr(case, fields[0]), label="Batch least squares")
        ax.semilogy(hours, getattr(case, fields[1]), label="EKF")
        decorate_plot(ax, case.label)
        ax.set_ylabel(ylabel)
    axes[0].legend()
    axes[0].set_title(f"{title} Error: {cases[0].label}")
    axes[1].set_xlabel("Time since start (hours)")
    fig.savefig(path, dpi=200)
    return fig


def plot_forecast_errors(cases, path):
    """Zoom into the held-out position forecasts to compare both estimators."""
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, constrained_layout=True
    )
    for ax, case in zip(axes, cases):
        hours = case.times_s / 3600.0
        forecast = case.times_s > FIT_END_S
        ax.plot(
            hours[forecast],
            case.batch_position_error_m[forecast],
            label="Batch least squares",
        )
        ax.plot(hours[forecast], case.ekf_position_error_m[forecast], label="EKF")
        ax.set_title(case.label)
        ax.set_ylabel("Position error (m)")
        ax.set_xlim(12.0, 24.0)
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_xlabel("Time since start (hours)")
    fig.suptitle("Held-Out Position Forecast (No Measurements After 12 h)")
    fig.savefig(path, dpi=200)
    return fig


def plot_ekf_uncertainty(cases, path):
    """Compare actual EKF position error and covariance-derived error scale.

    sqrt(trace(P_rr)) is a combined Cartesian one-sigma scale; it is not
    a calibrated 68%-confidence radius for the 3D position error norm.
    """
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7.6), sharex=True, constrained_layout=True
    )
    for ax, case in zip(axes, cases):
        hours = case.times_s / 3600.0
        ax.semilogy(
            hours,
            case.ekf_position_error_m,
            color="tab:blue",
            label="Actual EKF position error",
        )
        ax.semilogy(
            hours,
            case.ekf_position_uncertainty_scale_m,
            color="tab:purple",
            label=r"Estimated scale $\sqrt{\mathrm{tr}(P_{rr})}$",
        )
        decorate_plot(ax, case.label)
        ax.set_ylabel("Position scale (m)")
    axes[0].legend()
    axes[1].set_xlabel("Time since start (hours)")
    fig.suptitle("EKF Position Error Versus Predicted Uncertainty")
    fig.savefig(path, dpi=200)
    return fig


def print_results(cases):
    """Print forecast RMSE for each scenario and method."""
    print("\nHeld-out forecast: hours 12-24 (same data for both estimators)")
    print(
        f"{'Truth model':<22} {'Estimator':<22} "
        f"{'Position RMSE (m)':>18} {'Velocity RMSE (m/s)':>21}"
    )
    print("-" * 87)
    for case in cases:
        forecast = case.times_s > FIT_END_S
        for estimator, position, velocity in (
            (
                "Batch least squares",
                case.batch_position_error_m,
                case.batch_velocity_error_mps,
            ),
            ("EKF", case.ekf_position_error_m, case.ekf_velocity_error_mps),
        ):
            print(
                f"{case.label:<22} {estimator:<22} "
                f"{rms(position[forecast]):>18.2f} "
                f"{rms(velocity[forecast]):>21.5f}"
            )
        print(
            f"  Observations: {case.n_fit_measurements} fitted; "
            f"{case.n_heldout_measurements} withheld"
        )

    # A physics-only check: this is NOT an estimator error.
    position_difference = np.linalg.norm(
        cases[1].truth_states[-1, :3] - cases[0].truth_states[-1, :3]
    )
    print(
        f"\nDrag/no-drag true trajectory separation at 24 h: "
        f"{position_difference:.1f} m"
    )


def main():
    stations = example_stations()
    cases = [
        compare_one_truth_model(
            "Gravity + J2", OrbitalDynamics(include_j2=True), stations
        ),
        compare_one_truth_model(
            "Gravity + J2 + drag", DragAugmentedDynamics(), stations
        ),
    ]
    print_results(cases)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_errors(cases, "position", FIGURES_DIR / "drag_comparison_position.png")
    plot_errors(cases, "velocity", FIGURES_DIR / "drag_comparison_velocity.png")
    plot_forecast_errors(cases, FIGURES_DIR / "drag_comparison_forecast.png")
    plot_ekf_uncertainty(cases, FIGURES_DIR / "drag_comparison_uncertainty.png")
    print(f"Saved four figures to: {FIGURES_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
