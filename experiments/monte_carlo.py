"""Monte Carlo validation of batch orbit determination versus the EKF.

Run from the project root:
    python -m experiments.monte_carlo
    python -m experiments.monte_carlo --quick  # Fast pipeline check, not final data

We hold the true orbit and station geometry fixed, varying only independent
Gaussian range/range-rate measurement noise. Both estimators assume J2-only
gravity. The drag-affected truth model is used to calibrate the EKF process
noise on one group of seeds; disjoint seeds are reserved for final validation.

All scores use hours 12-24, when *neither* estimator receives measurements.
The calibration uses simulated ground truth, and is therefore a controlled
simulation study rather than an in-flight process-noise tuning method.
"""

import argparse
import csv
from dataclasses import replace
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

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CALIBRATION_SEEDS = tuple(range(100, 108))
VALIDATION_SEEDS = tuple(range(1000, 1012))
PSD_GRID = (1e-12, 1e-11, 1e-10, 1e-9, 1e-8)  # m^2/s^3
BASELINE_PSD = 1e-10
VARIABILITY_WEIGHT = 0.5
FIT_END_S = 12.0 * 3600.0
DURATION_S = 24.0 * 3600.0
MEASUREMENT_INTERVAL_S = 30.0
FORECAST_INTERVAL_S = 300.0
RANGE_STD_M = 10.0
RANGE_RATE_STD_MPS = 0.1
INITIAL_OFFSET = np.array([1000.0, -800.0, 500.0, 0.3, -0.2, 0.1])
PRIOR_COVARIANCE = np.diag([2000.0**2] * 3 + [1.0**2] * 3)


def validate_design(calibration_seeds, validation_seeds, psd_grid):
    """Prevent data leakage and duplicate experiments."""
    if not calibration_seeds or not validation_seeds or not psd_grid:
        raise ValueError("Both seed sets and the PSD grid must be nonempty")
    if len(set(calibration_seeds)) != len(calibration_seeds):
        raise ValueError("Duplicate calibration seeds")
    if len(set(validation_seeds)) != len(validation_seeds):
        raise ValueError("Duplicate validation seeds")
    if set(calibration_seeds) & set(validation_seeds):
        raise ValueError("Calibration and validation seeds must be disjoint")
    if len(set(psd_grid)) != len(psd_grid):
        raise ValueError("Duplicate PSD values")
    if any(not np.isfinite(q) or q < 0.0 for q in psd_grid):
        raise ValueError("All PSD values must be finite and nonnegative")


def noise_realization(ideal_measurements, seed):
    """Add noise without changing ideal ranges, visibility, or station geometry.

    The ideal measurement records come from a *noiseless* simulated trajectory.
    Real estimators see only the returned noisy measurements, not the truth.
    """
    rng = np.random.default_rng(seed)
    return [
        replace(
            measurement,
            range_m=measurement.range_m + rng.normal(0.0, RANGE_STD_M),
            range_rate_mps=(
                measurement.range_rate_mps + rng.normal(0.0, RANGE_RATE_STD_MPS)
            ),
            range_std_m=RANGE_STD_M,
            range_rate_std_mps=RANGE_RATE_STD_MPS,
        )
        for measurement in ideal_measurements
    ]


def scenario_truth(truth_dynamics, stations):
    """Simulate one fixed orbit and construct noiseless visible observations."""
    return simulate_tracking(
        initial_state=initial_orbit(),
        stations=stations,
        duration_s=DURATION_S,
        measurement_interval_s=MEASUREMENT_INTERVAL_S,
        dynamics=truth_dynamics,
        range_std_m=0.0,
        range_rate_std_mps=0.0,
        seed=0,
    )


def forecast_times():
    """Grid includes 12 h and 24 h, but RMSE excludes the 12-h boundary."""
    return np.arange(
        FIT_END_S, DURATION_S + 0.5 * FORECAST_INTERVAL_S, FORECAST_INTERVAL_S
    )


def truth_at_times(simulation, times_s):
    """Pick matching states without interpolating the high-rate truth grid."""
    indices = np.searchsorted(simulation.times_s, times_s)
    if np.any(indices >= len(simulation.times_s)) or not np.array_equal(
        simulation.times_s[indices], times_s
    ):
        raise ValueError("Forecast timestamps must lie on the truth grid")
    return simulation.true_states[indices]


def score_forecast(times_s, predicted_states, true_states, predicted_covariances=None):
    """Return 12-24-h position/velocity RMS and optional t=24 h position NEES.

    Position NEES = e_r^T P_rr^(-1) e_r, whose expectation is 3 for a
    correctly calibrated Gaussian 3D position estimate. This diagnostic is
    meaningful for the EKF only (batch LS supplies no state covariance).
    """
    times_s = np.asarray(times_s, dtype=float)
    predicted_states = np.asarray(predicted_states, dtype=float)
    true_states = np.asarray(true_states, dtype=float)
    if predicted_states.shape != (len(times_s), 6):
        raise ValueError("predicted_states must have shape (N, 6)")
    if true_states.shape != predicted_states.shape:
        raise ValueError("true_states must match predicted_states")
    mask = times_s > FIT_END_S
    if not np.any(mask) or not np.isclose(times_s[-1], DURATION_S):
        raise ValueError("The forecast must contain times after 12 h through 24 h")

    error = predicted_states - true_states
    position_squared = np.sum(error[mask, :3] ** 2, axis=1)
    velocity_squared = np.sum(error[mask, 3:] ** 2, axis=1)
    scores = {
        "forecast_position_rmse_m": float(np.sqrt(np.mean(position_squared))),
        "forecast_velocity_rmse_mps": float(np.sqrt(np.mean(velocity_squared))),
        "position_error_24h_m": float(np.linalg.norm(error[-1, :3])),
        "position_nees_24h": None,
        "predicted_position_scale_24h_m": None,
    }

    if predicted_covariances is not None:
        covariances = np.asarray(predicted_covariances, dtype=float)
        if covariances.shape != (len(times_s), 6, 6):
            raise ValueError("predicted_covariances must have shape (N, 6, 6)")
        p_rr = 0.5 * (covariances[-1, :3, :3] + covariances[-1, :3, :3].T)
        if np.min(np.linalg.eigvalsh(p_rr)) <= 0:
            raise ValueError("Position covariance at 24 h must be positive definite")
        scores["position_nees_24h"] = float(
            error[-1, :3] @ np.linalg.solve(p_rr, error[-1, :3])
        )
        scores["predicted_position_scale_24h_m"] = float(np.sqrt(np.trace(p_rr)))
    return scores


def evaluate_ekf(
    fitting_measurements, initial_guess, reference_truth, sample_times_s, q
):
    """Filter on 0-12 h, then forecast without any updates through 24 h."""
    ekf = run_ekf(
        measurements=fitting_measurements,
        initial_state=initial_guess,
        initial_covariance=PRIOR_COVARIANCE,
        output_times_s=sample_times_s,
        dynamics=OrbitalDynamics(include_j2=True),
        acceleration_psd=q,
    )
    indices = np.searchsorted(ekf.times_s, sample_times_s)
    if np.any(indices >= len(ekf.times_s)) or not np.array_equal(
        ekf.times_s[indices], sample_times_s
    ):
        raise RuntimeError("EKF did not return the requested forecast timestamps")
    return score_forecast(
        sample_times_s,
        ekf.states[indices],
        reference_truth,
        ekf.covariances[indices],
    )


def evaluate_batch(
    fitting_measurements, initial_guess, reference_truth, sample_times_s
):
    """Fit one initial state using 0-12 h, then propagate with J2 only."""
    batch = estimate_initial_state(
        measurements=fitting_measurements,
        initial_guess=initial_guess,
        dynamics=OrbitalDynamics(include_j2=True),
    )
    if not batch.success:
        raise RuntimeError(f"Batch least squares failed: {batch.message}")
    states = propagate_orbit(
        initial_state=batch.initial_state,
        t_span=(0.0, DURATION_S),
        t_eval=sample_times_s,
        dynamics=OrbitalDynamics(include_j2=True),
    ).y.T
    return score_forecast(sample_times_s, states, reference_truth)


def fitting_observations(ideal_measurements, seed):
    """Only measurements at or before the 12-h cutoff reach the estimators."""
    observations = noise_realization(ideal_measurements, seed)
    fitted = [m for m in observations if m.time_s <= FIT_END_S]
    if len(fitted) < 3:
        raise RuntimeError("Insufficient observations for batch estimation")
    return fitted


def select_psd(calibration_rows, variability_weight=VARIABILITY_WEIGHT):
    """Select q by minimizing mean RMSE plus a variability penalty."""
    psds = sorted({row["acceleration_psd"] for row in calibration_rows})

    if not psds:
        raise ValueError("Calibration rows cannot be empty")

    def objective(q):
        errors = np.array(
            [
                row["forecast_position_rmse_m"]
                for row in calibration_rows
                if row["acceleration_psd"] == q
            ]
        )

        mean = np.mean(errors)
        std = np.std(errors, ddof=1) if len(errors) > 1 else 0.0

        return mean + variability_weight * std

    return min(psds, key=lambda q: (objective(q), q))


def mean_std(values):
    """Report sample standard deviation, zero for a one-trial smoke test."""
    a = np.asarray(values, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0


def summarize_validation(validation_rows):
    """Aggregate only the genuinely unseen validation seeds."""
    summaries = []
    for scenario in ("Gravity + J2", "Gravity + J2 + drag"):
        for estimator in ("Batch least squares", "EKF baseline", "EKF tuned"):
            subset = [
                row
                for row in validation_rows
                if row["truth_model"] == scenario and row["estimator"] == estimator
            ]
            if not subset:
                continue
            pos_mean, pos_std = mean_std(
                [row["forecast_position_rmse_m"] for row in subset]
            )
            vel_mean, vel_std = mean_std(
                [row["forecast_velocity_rmse_mps"] for row in subset]
            )
            nees = [
                row["position_nees_24h"]
                for row in subset
                if row["position_nees_24h"] is not None
            ]
            summaries.append(
                {
                    "truth_model": scenario,
                    "estimator": estimator,
                    "n_trials": len(subset),
                    "position_rmse_mean_m": pos_mean,
                    "position_rmse_std_m": pos_std,
                    "velocity_rmse_mean_mps": vel_mean,
                    "velocity_rmse_std_mps": vel_std,
                    "position_nees_24h_mean": float(np.mean(nees)) if nees else None,
                }
            )
    return summaries


def write_csv(path, rows):
    if not rows:
        raise ValueError("No results to write")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_psd_calibration(
    rows, selected_psd, destination, variability_weight=VARIABILITY_WEIGHT
):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    psds = sorted({row["acceleration_psd"] for row in rows})

    means = []
    objectives = []

    for q in psds:
        values = np.array(
            [
                row["forecast_position_rmse_m"]
                for row in rows
                if row["acceleration_psd"] == q
            ]
        )

        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0

        means.append(mean)
        objectives.append(mean + variability_weight * std)

        ax.scatter(
            [q] * len(values),
            values,
            alpha=0.35,
            color="tab:blue",
            s=22,
        )

    # Mean RMSE
    ax.plot(
        psds,
        means,
        color="tab:blue",
        marker="o",
        lw=2,
        label="Mean RMSE",
    )

    # Objective function
    ax.plot(
        psds,
        objectives,
        color="tab:green",
        marker="s",
        lw=2,
        label=f"Mean RMSE + {variability_weight} × SD",
    )

    # Selected process noise
    ax.axvline(
        selected_psd,
        color="tab:orange",
        ls="--",
        label=f"Selected q = {selected_psd:.0e}",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"EKF acceleration PSD $q$ (m$^2$/s$^3$)")
    ax.set_ylabel("RMSE / Objective (m)")
    ax.set_title("Calibration on Drag-Affected Truth")

    ax.grid(alpha=0.25)
    ax.legend()

    fig.savefig(destination, dpi=200)
    plt.close(fig)


def plot_validation(validation_rows, destination):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    estimator_names = ("Batch least squares", "EKF baseline", "EKF tuned")
    for ax, scenario in zip(axes, ("Gravity + J2", "Gravity + J2 + drag")):
        means = []
        stds = []
        for name in estimator_names:
            values = [
                row["forecast_position_rmse_m"]
                for row in validation_rows
                if row["truth_model"] == scenario and row["estimator"] == name
            ]
            mu, sigma = mean_std(values)
            means.append(mu)
            stds.append(sigma)
        x = np.arange(len(estimator_names))
        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            color=("tab:blue", "tab:orange", "tab:green"),
            alpha=0.85,
        )
        ax.set_xticks(x, ["Batch LS", "EKF baseline", "EKF tuned"])
        ax.set_ylabel("Mean held-out position RMSE (m)")
        ax.set_title(scenario)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Validation Seeds: Forecast Hours 12-24 (Bars = Mean ± Sample SD)")
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def run_study(
    calibration_seeds=CALIBRATION_SEEDS,
    validation_seeds=VALIDATION_SEEDS,
    psd_grid=PSD_GRID,
    results_dir=RESULTS_DIR,
):
    """Run calibration and disjoint validation, writing CSVs and two figures."""
    validate_design(calibration_seeds, validation_seeds, psd_grid)
    results_dir = Path(results_dir)
    raw_dir = results_dir / "raw"
    figures_dir = results_dir / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    stations = example_stations()
    sample_times = forecast_times()
    initial_guess = initial_orbit() + INITIAL_OFFSET

    drag_simulation = scenario_truth(DragAugmentedDynamics(), stations)
    drag_truth = truth_at_times(drag_simulation, sample_times)
    calibration_rows = []
    total_calibration = len(calibration_seeds) * len(psd_grid)
    print(
        f"Calibrating on {len(calibration_seeds)} drag-affected seeds "
        f"× {len(psd_grid)} q values ({total_calibration} EKF fits)...",
        flush=True,
    )
    for i, seed in enumerate(calibration_seeds, start=1):
        fitted = fitting_observations(drag_simulation.measurements, seed)
        for q in psd_grid:
            scores = evaluate_ekf(fitted, initial_guess, drag_truth, sample_times, q)
            calibration_rows.append(
                {
                    "seed": seed,
                    "acceleration_psd": q,
                    **scores,
                }
            )
        print(f"  Calibration seed {i}/{len(calibration_seeds)} complete", flush=True)

    selected_psd = select_psd(
        calibration_rows,
        variability_weight=VARIABILITY_WEIGHT,
    )
    print(f"Chosen EKF acceleration PSD: {selected_psd:.2e} m^2/s^3", flush=True)

    no_drag_simulation = scenario_truth(OrbitalDynamics(include_j2=True), stations)
    scenarios = (
        ("Gravity + J2", no_drag_simulation),
        ("Gravity + J2 + drag", drag_simulation),
    )
    validation_rows = []
    for scenario_label, simulation in scenarios:
        truth = truth_at_times(simulation, sample_times)
        print(
            f"Validating: {scenario_label} ({len(validation_seeds)} seeds)...",
            flush=True,
        )
        for i, seed in enumerate(validation_seeds, start=1):
            fitted = fitting_observations(simulation.measurements, seed)
            batch_scores = evaluate_batch(fitted, initial_guess, truth, sample_times)
            common = {
                "truth_model": scenario_label,
                "seed": seed,
                "n_fitted_observations": len(fitted),
            }
            validation_rows.append(
                {
                    **common,
                    "estimator": "Batch least squares",
                    "acceleration_psd": None,
                    **batch_scores,
                }
            )
            # Avoid redundant computation when the tuned q equals the baseline.
            ekf_scores_by_q = {}
            for q in (BASELINE_PSD, selected_psd):
                if q not in ekf_scores_by_q:
                    ekf_scores_by_q[q] = evaluate_ekf(
                        fitted, initial_guess, truth, sample_times, q
                    )
            for name, q in (
                ("EKF baseline", BASELINE_PSD),
                ("EKF tuned", selected_psd),
            ):
                validation_rows.append(
                    {
                        **common,
                        "estimator": name,
                        "acceleration_psd": q,
                        **ekf_scores_by_q[q],
                    }
                )
            print(f"  Validation seed {i}/{len(validation_seeds)} complete", flush=True)

    summary_rows = summarize_validation(validation_rows)
    write_csv(raw_dir / "monte_carlo_calibration.csv", calibration_rows)
    write_csv(raw_dir / "monte_carlo_validation.csv", validation_rows)
    write_csv(raw_dir / "monte_carlo_summary.csv", summary_rows)
    if all(q > 0 for q in psd_grid):
        plot_psd_calibration(
            calibration_rows,
            selected_psd,
            figures_dir / "monte_carlo_psd_sensitivity.png",
            variability_weight=VARIABILITY_WEIGHT,
        )
    plot_validation(validation_rows, figures_dir / "monte_carlo_validation.png")

    print("\nHeld-out position RMSE across VALIDATION noise seeds (m):")
    print(
        f"{'Truth':<23} {'Estimator':<22} {'Mean':>12} {'Sample SD':>12} "
        f"{'Mean NEES at 24 h':>19}"
    )
    print("-" * 91)
    for row in summary_rows:
        nees = (
            f"{row['position_nees_24h_mean']:.1f}"
            if row["position_nees_24h_mean"] is not None
            else "—"
        )
        print(
            f"{row['truth_model']:<23} {row['estimator']:<22} "
            f"{row['position_rmse_mean_m']:>12.2f} "
            f"{row['position_rmse_std_m']:>12.2f} {nees:>19}"
        )
    print("Expected mean 3D position NEES ≈ 3 for a calibrated EKF.")
    print(
        "Only measurement noise varies across seeds; the orbit, drag and "
        "station geometry are fixed."
    )
    print(f"Saved three CSVs in {raw_dir} and two plots in {figures_dir}")
    return selected_psd, calibration_rows, validation_rows, summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run 2 calibration and 3 validation seeds for a pipeline check",
    )
    options = parser.parse_args()
    if options.quick:
        print("QUICK CHECK.", flush=True)
    run_study(
        calibration_seeds=CALIBRATION_SEEDS[:2] if options.quick else CALIBRATION_SEEDS,
        validation_seeds=VALIDATION_SEEDS[:3] if options.quick else VALIDATION_SEEDS,
        psd_grid=PSD_GRID[1:4] if options.quick else PSD_GRID,
        results_dir=RESULTS_DIR / "quick_check" if options.quick else RESULTS_DIR,
    )


if __name__ == "__main__":
    main()
