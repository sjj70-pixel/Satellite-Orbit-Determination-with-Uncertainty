"""Visualize a 24-hour synthetic satellite tracking mission.

Run from the project root:
    python -m experiments.tracking_demo

The three ground stations are illustrative, not actual tracking facilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import MU_EARTH, R_EARTH
from src.ground_stations import GroundStation
from src.simulation import simulate_tracking

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
DATA_DIR = PROJECT_ROOT / "results" / "raw"


def initial_orbit():
    """A 500 km circular orbit inclined 55 degrees to the equator."""
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


def example_stations():
    """Three geographically separated, hypothetical tracking stations."""
    return [
        GroundStation("Station A", latitude_deg=0.0, longitude_deg=0.0),
        GroundStation("Station B", latitude_deg=45.0, longitude_deg=80.0),
        GroundStation("Station C", latitude_deg=-35.0, longitude_deg=-120.0),
    ]


def plot_measurements(measurements, output_path):
    """Plot available range and range-rate observations by station."""
    df = pd.DataFrame(measurements)
    if df.empty:
        raise ValueError("No measurements were generated; check station visibility")

    fig, (ax_range, ax_rate) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, constrained_layout=True
    )

    for station_name, group in df.groupby("station", sort=False):
        hours = group["time_s"] / 3600.0
        ax_range.scatter(
            hours,
            group["range_m"] / 1000.0,
            s=7,
            alpha=0.8,
            label=station_name,
        )
        ax_rate.scatter(
            hours,
            group["range_rate_mps"],
            s=7,
            alpha=0.8,
            label=station_name,
        )

    ax_range.set_title("Ground-Station Measurements (24 Hours)")
    ax_range.set_ylabel("Range (km)")
    ax_rate.set_ylabel("Range rate (m/s)")
    ax_rate.set_xlabel("Time since start (hours)")
    ax_range.legend(markerscale=2.0)

    for ax in (ax_range, ax_rate):
        ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=200)
    return fig


def plot_visibility(simulation, stations, interval_s, output_path):
    """Show when the satellite is above each station's elevation mask."""
    visible = np.array(
        [
            [
                station.is_visible(state[:3], t)
                for t, state in zip(simulation.times_s, simulation.true_states)
            ]
            for station in stations
        ],
        dtype=int,
    )

    hours = simulation.times_s / 3600.0
    half_bin_hours = interval_s / 7200.0

    fig, ax = plt.subplots(figsize=(11, 3.4), constrained_layout=True)
    ax.imshow(
        visible,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="Greens",
        vmin=0,
        vmax=1,
        extent=(
            hours[0] - half_bin_hours,
            hours[-1] + half_bin_hours,
            -0.5,
            len(stations) - 0.5,
        ),
    )

    ax.set_yticks(range(len(stations)), [station.name for station in stations])
    ax.set_xlabel("Time since start (hours)")
    ax.set_title("Tracking Availability (Green = Visible)")
    ax.set_xlim(0, 24)

    fig.savefig(output_path, dpi=200)
    return fig


def main():
    duration_s = 24 * 3600.0
    interval_s = 30.0
    stations = example_stations()

    simulation = simulate_tracking(
        initial_state=initial_orbit(),
        stations=stations,
        duration_s=duration_s,
        measurement_interval_s=interval_s,
        range_std_m=10.0,
        range_rate_std_mps=0.1,
        seed=42,
    )

    measurements = simulation.measurement_rows()
    df = pd.DataFrame(measurements)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / "tracking_observations.csv"
    df.to_csv(csv_path, index=False)

    plot_measurements(measurements, FIGURES_DIR / "tracking_measurements.png")
    plot_visibility(
        simulation,
        stations,
        interval_s,
        FIGURES_DIR / "tracking_visibility.png",
    )

    print(f"Trajectory samples: {len(simulation.times_s):,}")
    print(f"Total observations: {len(df):,}")
    print("Observations by station:")
    print(df.groupby("station").size().to_string())
    print("\nFirst five observations:")
    print(df.head().to_string(index=False))
    print(f"\nSaved observations: {csv_path}")
    print(f"Saved figures: {FIGURES_DIR}")

    plt.show()


if __name__ == "__main__":
    main()
