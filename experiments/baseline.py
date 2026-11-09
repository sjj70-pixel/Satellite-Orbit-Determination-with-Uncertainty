"""
Baseline experiments for orbital propagation.

Experiments:
    1. Circular two-body orbit
    2. Elliptical two-body orbit
    3. Comparison of two-body and J2 propagation

All calculations use SI units.
Plots display distances in kilometers.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src.constants import MU_EARTH, R_EARTH
from src.dynamics import OrbitalDynamics
from src.propagation import propagate_orbit

# Directory for generated figures
FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"


def orbital_period(semi_major_axis):
    """Calculate the Keplerian orbital period (seconds)."""
    return 2 * np.pi * np.sqrt(semi_major_axis**3 / MU_EARTH)


def circular_initial_state(altitude, inclination=0.0):
    """
    Construct a circular two-body orbit initial state.

    Parameters
    ----------
    altitude : float
        Altitude above Earth's equatorial radius (m).
    inclination : float
        Orbital inclination in radians.
    """
    radius = R_EARTH + altitude
    speed = np.sqrt(MU_EARTH / radius)

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


def plot_orbit_xy(ax, solution, title):
    """Plot an orbital trajectory projected onto the XY plane."""
    x = solution.y[0] / 1000
    y = solution.y[1] / 1000

    earth = Circle((0, 0), R_EARTH / 1000, color="steelblue", alpha=0.5, label="Earth")

    ax.add_patch(earth)

    ax.plot(x, y, label="Satellite trajectory")
    ax.scatter(x[0], y[0], color="green", s=35, label="Initial position")

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()


def save_figure(fig, filename):
    """Save a figure in the results directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=200)


def circular_orbit_experiment():
    """Simulate one complete circular orbit."""
    altitude = 500_000.0

    initial_state = circular_initial_state(altitude)
    radius = R_EARTH + altitude
    period = orbital_period(radius)

    solution = propagate_orbit(
        initial_state,
        t_span=(0, period),
        dynamics=OrbitalDynamics(include_j2=False),
        t_eval=np.linspace(0, period, 800),
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    plot_orbit_xy(ax, solution, "Circular Orbit (500 km Altitude)")

    fig.tight_layout()
    save_figure(fig, "circular_orbit.png")

    radii = np.linalg.norm(solution.y[:3], axis=0)

    print("\nCircular Orbit")
    print(f"Orbital period: {period / 60:.2f} minutes")
    print(f"Maximum radius variation: {np.ptp(radii):.6f} m")


def elliptical_orbit_experiment():
    """Simulate one complete elliptical orbit."""
    rp = R_EARTH + 500_000.0
    ra = R_EARTH + 1_200_000.0

    semi_major_axis = (rp + ra) / 2

    # Vis-viva equation at perigee
    vp = np.sqrt(MU_EARTH * (2 / rp - 1 / semi_major_axis))

    initial_state = np.array([rp, 0.0, 0.0, 0.0, vp, 0.0])

    period = orbital_period(semi_major_axis)

    solution = propagate_orbit(
        initial_state,
        t_span=(0, period),
        dynamics=OrbitalDynamics(include_j2=False),
        t_eval=np.linspace(0, period, 800),
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    plot_orbit_xy(ax, solution, "Elliptical Orbit")

    fig.tight_layout()
    save_figure(fig, "elliptical_orbit.png")

    radii = np.linalg.norm(solution.y[:3], axis=0)

    print("\nElliptical Orbit")
    print(f"Orbital period: {period / 60:.2f} minutes")
    print(f"Minimum altitude: {(radii.min() - R_EARTH) / 1000:.2f} km")
    print(f"Maximum altitude: {(radii.max() - R_EARTH) / 1000:.2f} km")


def j2_comparison_experiment():
    """Compare two-body and J2 propagation over 48 hours."""
    altitude = 500_000.0
    inclination = np.radians(55.0)

    initial_state = circular_initial_state(altitude, inclination)

    duration = 48 * 3600
    times = np.linspace(0, duration, 1500)

    two_body_solution = propagate_orbit(
        initial_state,
        t_span=(0, duration),
        dynamics=OrbitalDynamics(include_j2=False),
        t_eval=times,
    )

    j2_solution = propagate_orbit(
        initial_state,
        t_span=(0, duration),
        dynamics=OrbitalDynamics(include_j2=True),
        t_eval=times,
    )

    # Position difference at the same times
    position_difference = j2_solution.y[:3] - two_body_solution.y[:3]

    separation = np.linalg.norm(position_difference, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(times / 3600, separation / 1000)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Position difference (km)")
    ax.set_title("Effect of J2 on Orbital Propagation")
    ax.grid(True)

    fig.tight_layout()
    save_figure(fig, "j2_comparison.png")

    print("\nJ2 Comparison")
    print(f"Final position difference: {separation[-1] / 1000:.2f} km")
    print(f"Maximum position difference: {separation.max() / 1000:.2f} km")


def main():
    circular_orbit_experiment()
    elliptical_orbit_experiment()
    j2_comparison_experiment()

    plt.show()


if __name__ == "__main__":
    main()
