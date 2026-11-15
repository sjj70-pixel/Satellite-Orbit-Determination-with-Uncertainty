"""To match the batch least-squares baseline, the EKF uses observations
from only the first 12 hours. The remaining 12 hours are withheld for
forecasting, although ground-station visibility is still displayed.

Truth and estimator both use two-body + J2 dynamics in this baseline.
The effects of unmodeled atmospheric drag are examined separately in
drag_comparison.py and monte_carlo.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.dynamics import OrbitalDynamics
from src.ekf import run_ekf
from src.propagation import propagate_orbit
from src.simulation import simulate_tracking

from .tracking_demo import example_stations, initial_orbit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

DURATION_S = 24.0 * 3600.0
FIT_DURATION_S = 12.0 * 3600.0
MEASUREMENT_INTERVAL_S = 30.0
OUTPUT_INTERVAL_S = 300.0
ACCELERATION_PSD = 1e-10  # White-acceleration noise (m^2 / s^3).


def plot_error_norms(hours, initial_error, filtered_error, ylabel, title, path):
    """Plot the magnitude of the initial-guess and EKF errors."""
    fig, ax = plt.subplots(figsize=(11, 4.4), constrained_layout=True)
    ax.semilogy(
        hours, initial_error, label="Uncorrected initial guess", color="tab:orange"
    )
    ax.semilogy(hours, filtered_error, label="EKF estimate", color="tab:blue")
    ax.axvline(FIT_DURATION_S / 3600.0, color="black", linestyle="--", lw=1)
    ax.axvspan(
        FIT_DURATION_S / 3600.0,
        DURATION_S / 3600.0,
        color="gray",
        alpha=0.12,
        label="Held-out forecast",
    )
    ax.set(title=title, xlabel="Time since start (hours)", ylabel=ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=200)
    return fig


def plot_position_uncertainty(hours, position_errors, covariances, path):
    """Plot each ECI position error against its marginal +/-3-sigma envelope.

    These bands refer to individual Cartesian components, NOT a three-
    dimensional 99.7%-probability ball. The symmetric logarithmic axes make
    both the large initial errors and small later errors visible.
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 8.2), sharex=True, constrained_layout=True
    )
    labels = ["x", "y", "z"]
    for j, (ax, label) in enumerate(zip(axes, labels)):
        sigma = np.sqrt(np.maximum(covariances[:, j, j], 0.0))
        band = 3.0 * sigma
        ax.fill_between(hours, -band, band, color="tab:green", alpha=0.24, label="±3σ")
        ax.plot(
            hours,
            position_errors[:, j],
            color="tab:blue",
            lw=1.3,
            label="Actual estimation error",
        )
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.axvline(FIT_DURATION_S / 3600.0, color="black", linestyle="--", lw=1)
        ax.axvspan(
            FIT_DURATION_S / 3600.0, DURATION_S / 3600.0, color="gray", alpha=0.08
        )
        ax.set_yscale("symlog", linthresh=10.0)
        ax.set_ylabel(f"ECI {label} error (m)")
        ax.grid(True, which="both", alpha=0.2)

    axes[0].set_title("EKF Cartesian Position Error and ±3σ Uncertainty")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time since start (hours)")
    fig.savefig(path, dpi=200)
    return fig


def plot_visibility_and_covariance(
    simulation, stations, ekf_result, position_error_m, path
):
    """Show ground-station passes and estimated uncertainty through the gaps."""
    fig, (ax_passes, ax_cov) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 2.0]},
    )

    for index, station in enumerate(stations):
        station_observations = [
            measurement
            for measurement in simulation.measurements
            if measurement.station.name == station.name
        ]
        used = np.array(
            [
                measurement.time_s <= FIT_DURATION_S
                for measurement in station_observations
            ]
        )
        times = np.array(
            [measurement.time_s / 3600.0 for measurement in station_observations]
        )
        ax_passes.scatter(
            times[used],
            np.full(np.sum(used), index),
            marker="|",
            s=180,
            color="tab:green",
            linewidth=1.4,
            label="Used by EKF" if index == 0 else None,
        )
        ax_passes.scatter(
            times[~used],
            np.full(np.sum(~used), index),
            marker="|",
            s=180,
            color="gray",
            linewidth=1.4,
            label="Withheld" if index == 0 else None,
        )

    ax_passes.set_yticks(range(len(stations)), [s.name for s in stations])
    ax_passes.set_ylim(-0.6, len(stations) - 0.4)
    ax_passes.set_title("Ground-Station Observations and EKF Position Uncertainty")
    ax_passes.legend(loc="upper right")
    ax_passes.grid(True, axis="x", alpha=0.25)

    hours = ekf_result.times_s / 3600.0
    # sqrt(trace(P_rr)) is the combined Cartesian 1-sigma error scale.
    # It is not a 68% confidence radius for the three-dimensional error norm.
    position_scale = np.sqrt(
        np.maximum(np.trace(ekf_result.covariances[:, :3, :3], axis1=1, axis2=2), 0.0)
    )
    ax_cov.semilogy(
        hours,
        position_scale,
        color="tab:purple",
        label=r"Estimated scale $\sqrt{\mathrm{tr}(P_{rr})}$",
    )
    ax_cov.semilogy(
        hours,
        position_error_m,
        color="tab:blue",
        alpha=0.8,
        label="Actual position error",
    )
    has_update = ekf_result.update_counts > 0
    ax_cov.scatter(
        hours[has_update],
        position_scale[has_update],
        s=9,
        color="tab:green",
        zorder=3,
        label="Measurement updates",
    )
    for ax in (ax_passes, ax_cov):
        ax.axvline(FIT_DURATION_S / 3600.0, color="black", linestyle="--", lw=1)
        ax.axvspan(
            FIT_DURATION_S / 3600.0, DURATION_S / 3600.0, color="gray", alpha=0.1
        )
    ax_cov.set(xlabel="Time since start (hours)", ylabel="Position scale (m)")
    ax_cov.grid(True, which="both", alpha=0.2)
    ax_cov.legend(loc="upper right")
    ax_cov.set_xlim(0.0, 24.0)

    fig.savefig(path, dpi=200)
    return fig


def rms(values):
    """Root-mean-square of the entries of a one-dimensional array."""
    return float(np.sqrt(np.mean(np.square(values))))


def main():
    stations = example_stations()
    true_initial_state = initial_orbit()
    gravity = OrbitalDynamics(include_j2=True)

    simulation = simulate_tracking(
        initial_state=true_initial_state,
        stations=stations,
        duration_s=DURATION_S,
        measurement_interval_s=MEASUREMENT_INTERVAL_S,
        dynamics=gravity,
        range_std_m=10.0,
        range_rate_std_mps=0.1,
        seed=42,
    )
    observations_used = [
        observation
        for observation in simulation.measurements
        if observation.time_s <= FIT_DURATION_S
    ]
    observations_withheld = [
        observation
        for observation in simulation.measurements
        if observation.time_s > FIT_DURATION_S
    ]
    if not observations_used or not observations_withheld:
        raise RuntimeError("Expected measurements in both 12-hour periods")

    # Same initial state error as the batch-estimation benchmark.
    initial_guess = true_initial_state + np.array(
        [1000.0, -800.0, 500.0, 0.3, -0.2, 0.1]
    )
    initial_covariance = np.diag([2000.0**2] * 3 + [1.0**2] * 3)

    ekf = run_ekf(
        measurements=observations_used,
        initial_state=initial_guess,
        initial_covariance=initial_covariance,
        output_times_s=np.arange(
            0.0, DURATION_S + OUTPUT_INTERVAL_S, OUTPUT_INTERVAL_S
        ),
        dynamics=gravity,
        acceleration_psd=ACCELERATION_PSD,
    )

    # Ground truth is used ONLY for evaluation and plotting, never by run_ekf.
    truth_indices = np.searchsorted(simulation.times_s, ekf.times_s)
    if np.any(truth_indices >= len(simulation.times_s)) or not np.allclose(
        simulation.times_s[truth_indices], ekf.times_s, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError("EKF output times must lie on the truth sampling grid")
    truth_at_ekf_times = simulation.true_states[truth_indices]

    initial_trajectory = propagate_orbit(
        initial_state=initial_guess,
        t_span=(0.0, DURATION_S),
        t_eval=ekf.times_s,
        dynamics=gravity,
    ).y.T

    errors = ekf.states - truth_at_ekf_times
    position_error = np.linalg.norm(errors[:, :3], axis=1)
    velocity_error = np.linalg.norm(errors[:, 3:], axis=1)
    initial_errors = initial_trajectory - truth_at_ekf_times
    initial_position_error = np.linalg.norm(initial_errors[:, :3], axis=1)
    initial_velocity_error = np.linalg.norm(initial_errors[:, 3:], axis=1)
    hours = ekf.times_s / 3600.0

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_error_norms(
        hours,
        initial_position_error,
        position_error,
        "Position error (m)",
        "EKF Satellite Position Error",
        FIGURES_DIR / "ekf_position_error.png",
    )
    plot_error_norms(
        hours,
        initial_velocity_error,
        velocity_error,
        "Velocity error (m/s)",
        "EKF Satellite Velocity Error",
        FIGURES_DIR / "ekf_velocity_error.png",
    )
    plot_position_uncertainty(
        hours,
        errors[:, :3],
        ekf.covariances,
        FIGURES_DIR / "ekf_position_uncertainty.png",
    )
    plot_visibility_and_covariance(
        simulation,
        stations,
        ekf,
        position_error,
        FIGURES_DIR / "ekf_visibility_uncertainty.png",
    )

    during_fit = ekf.times_s <= FIT_DURATION_S
    during_forecast = ekf.times_s > FIT_DURATION_S
    last_fit_index = np.flatnonzero(during_fit)[-1]
    fit_sigma_position = np.sqrt(np.trace(ekf.covariances[last_fit_index, :3, :3]))
    final_sigma_position = np.sqrt(np.trace(ekf.covariances[-1, :3, :3]))

    print(f"Observations supplied to EKF: {len(observations_used)}")
    print(f"Withheld observations: {len(observations_withheld)}")
    print(f"Initial position error: {initial_position_error[0]:.2f} m")
    print(f"Position error at 12 h: {position_error[last_fit_index]:.2f} m")
    print(f"Position error at 24 h: {position_error[-1]:.2f} m")
    print(f"RMS position error during fit: {rms(position_error[during_fit]):.2f} m")
    print(
        f"RMS position error during held-out forecast: "
        f"{rms(position_error[during_forecast]):.2f} m"
    )
    print(
        f"RMS velocity error during held-out forecast: "
        f"{rms(velocity_error[during_forecast]):.5f} m/s"
    )
    print(
        f"Estimated position scale at 12 h (sqrt(trace(P_rr))): "
        f"{fit_sigma_position:.2f} m"
    )
    print(
        f"Estimated position scale at 24 h (sqrt(trace(P_rr))): "
        f"{final_sigma_position:.2f} m"
    )
    print(f"Saved four figures to: {FIGURES_DIR}")

    plt.show()


if __name__ == "__main__":
    main()
