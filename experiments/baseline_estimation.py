"""Evaluate batch least-squares orbit determination on synthetic tracking data.

Run from the project root:
    python -m experiments.baseline_estimation

We fit only observations from the first 12 hours of a 24-hour simulation.
The final 12 hours are withheld to test the estimated orbit's forecast.
Both truth and estimator use two-body + J2 dynamics in this baseline test;
future experiments will introduce unmodeled drag into the truth dynamics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.batch_estimator import estimate_initial_state
from src.dynamics import OrbitalDynamics
from src.measurements import predict_measurement
from src.propagation import propagate_orbit
from src.simulation import simulate_tracking

from .tracking_demo import example_stations, initial_orbit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def plot_state_error(
    hours, initial_error, fitted_error, fit_hours, ylabel, title, path
):
    """Compare uncorrected and batch-fitted orbit errors on a log scale."""
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.semilogy(hours, initial_error, color="tab:orange", label="Initial guess")
    ax.semilogy(hours, fitted_error, color="tab:blue", label="Batch estimate")
    ax.axvline(fit_hours, color="black", linestyle="--", linewidth=1)
    ax.axvspan(
        fit_hours, hours[-1], color="gray", alpha=0.12, label="Held-out forecast"
    )
    ax.set(title=title, xlabel="Time since start (hours)", ylabel=ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=200)
    return fig


def normalized_residuals(measurements, sample_times, estimated_states):
    """Evaluate observed-minus-predicted errors in measurement-sigma units.

    Returns rows of [range residual / sigma, range-rate residual / sigma].
    These diagnostics include held-out observations, which were NOT used
    when fitting the estimated initial state.
    """
    residuals = np.empty((len(measurements), 2), dtype=float)

    for i, observation in enumerate(measurements):
        index = int(np.searchsorted(sample_times, observation.time_s))
        if index >= len(sample_times) or not np.isclose(
            sample_times[index], observation.time_s, rtol=0.0, atol=1e-7
        ):
            raise ValueError("Observation time is not in the simulation time grid")

        predicted = predict_measurement(
            estimated_states[index], observation.station, observation.time_s
        )
        # Positive defaults are used in this experiment. The numerical floors
        # also make this helper consistent with the batch estimator.
        sigma = np.maximum(
            [observation.range_std_m, observation.range_rate_std_mps],
            [1.0, 0.01],
        )
        residuals[i] = (observation.vector() - predicted) / sigma

    return residuals


def plot_measurement_residuals(measurements, residuals, fit_hours, path):
    """Plot normalized range and range-rate residuals for each station."""
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, constrained_layout=True
    )
    station_names = list(dict.fromkeys(m.station.name for m in measurements))
    times_hours = np.array([m.time_s for m in measurements]) / 3600.0

    for name in station_names:
        indices = np.array([m.station.name == name for m in measurements])
        for column, ax in enumerate(axes):
            ax.scatter(
                times_hours[indices],
                residuals[indices, column],
                s=12,
                alpha=0.75,
                label=name,
            )

    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axhline(3.0, color="gray", linestyle=":", linewidth=1)
        ax.axhline(-3.0, color="gray", linestyle=":", linewidth=1)
        ax.axvline(fit_hours, color="black", linestyle="--", linewidth=1)
        ax.axvspan(fit_hours, 24.0, color="gray", alpha=0.12)
        ax.grid(True, alpha=0.2)

    axes[0].set(
        title="Measurement Residuals (First 12 h Fit; Last 12 h Held Out)",
        ylabel="Range residual / sigma",
    )
    axes[1].set(xlabel="Time since start (hours)", ylabel="Range-rate residual / sigma")
    axes[0].legend(markerscale=1.5)
    fig.savefig(path, dpi=200)
    return fig


def main():
    duration_s = 24.0 * 3600.0
    fit_duration_s = 12.0 * 3600.0
    stations = example_stations()
    true_initial_state = initial_orbit()
    dynamics = OrbitalDynamics(include_j2=True)

    simulation = simulate_tracking(
        initial_state=true_initial_state,
        stations=stations,
        duration_s=duration_s,
        measurement_interval_s=30.0,
        dynamics=dynamics,
        range_std_m=10.0,
        range_rate_std_mps=0.1,
        seed=42,
    )

    # This offset emulates an imperfect orbital state supplied to the estimator.
    # It is used to build the starting guess, never by the least-squares solver.
    initial_guess = true_initial_state + np.array(
        [1000.0, -800.0, 500.0, 0.3, -0.2, 0.1]
    )

    training = [m for m in simulation.measurements if m.time_s <= fit_duration_s]
    held_out = [m for m in simulation.measurements if m.time_s > fit_duration_s]
    if len(training) < 3 or not held_out:
        raise RuntimeError(
            "Not enough tracking observations in the fitting or forecast period"
        )

    estimate = estimate_initial_state(training, initial_guess, dynamics=dynamics)
    if not estimate.success:
        raise RuntimeError(f"Batch estimator did not converge: {estimate.message}")

    def trajectory_from(initial_state):
        return propagate_orbit(
            initial_state=initial_state,
            t_span=(0.0, duration_s),
            t_eval=simulation.times_s,
            dynamics=dynamics,
        ).y.T

    initial_trajectory = trajectory_from(initial_guess)
    fitted_trajectory = trajectory_from(estimate.initial_state)

    initial_position_error = np.linalg.norm(
        initial_trajectory[:, :3] - simulation.true_states[:, :3], axis=1
    )
    fitted_position_error = np.linalg.norm(
        fitted_trajectory[:, :3] - simulation.true_states[:, :3], axis=1
    )
    initial_velocity_error = np.linalg.norm(
        initial_trajectory[:, 3:] - simulation.true_states[:, 3:], axis=1
    )
    fitted_velocity_error = np.linalg.norm(
        fitted_trajectory[:, 3:] - simulation.true_states[:, 3:], axis=1
    )

    residuals = normalized_residuals(
        simulation.measurements, simulation.times_s, fitted_trajectory
    )
    in_fit = np.array([m.time_s <= fit_duration_s for m in simulation.measurements])
    hours = simulation.times_s / 3600.0
    in_forecast = simulation.times_s > fit_duration_s

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_state_error(
        hours,
        initial_position_error,
        fitted_position_error,
        12.0,
        "Position error (m)",
        "Satellite Position Error",
        FIGURES_DIR / "batch_position_error.png",
    )
    plot_state_error(
        hours,
        initial_velocity_error,
        fitted_velocity_error,
        12.0,
        "Velocity error (m/s)",
        "Satellite Velocity Error",
        FIGURES_DIR / "batch_velocity_error.png",
    )
    plot_measurement_residuals(
        simulation.measurements,
        residuals,
        12.0,
        FIGURES_DIR / "batch_measurement_residuals.png",
    )

    initial_state_position_error = np.linalg.norm(
        initial_guess[:3] - true_initial_state[:3]
    )
    estimated_state_position_error = np.linalg.norm(
        estimate.initial_state[:3] - true_initial_state[:3]
    )
    initial_state_velocity_error = np.linalg.norm(
        initial_guess[3:] - true_initial_state[3:]
    )
    estimated_state_velocity_error = np.linalg.norm(
        estimate.initial_state[3:] - true_initial_state[3:]
    )

    print(f"Observations used for fitting: {len(training)}")
    print(f"Held-out observations: {len(held_out)}")
    print(f"Optimization evaluations: {estimate.nfev}")
    print(f"Initial position error at t=0: {initial_state_position_error:.2f} m")
    print(f"Estimated position error at t=0: {estimated_state_position_error:.2f} m")
    print(f"Initial velocity error at t=0: {initial_state_velocity_error:.5f} m/s")
    print(f"Estimated velocity error at t=0: {estimated_state_velocity_error:.5f} m/s")
    print(
        f"RMS position error over held-out 12 h: "
        f"{np.sqrt(np.mean(fitted_position_error[in_forecast]**2)):.2f} m"
    )
    print(
        f"RMS velocity error over held-out 12 h: "
        f"{np.sqrt(np.mean(fitted_velocity_error[in_forecast]**2)):.5f} m/s"
    )
    for label, mask in (("Fit", in_fit), ("Held-out", ~in_fit)):
        rms = np.sqrt(np.mean(residuals[mask] ** 2, axis=0))
        print(
            f"{label} normalized RMS residuals: "
            f"range={rms[0]:.2f}, range-rate={rms[1]:.2f}"
        )
    print(f"Saved three figures to: {FIGURES_DIR}")

    plt.show()


if __name__ == "__main__":
    main()
