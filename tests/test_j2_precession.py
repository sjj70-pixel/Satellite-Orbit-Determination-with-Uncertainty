"""
Verify J2-induced nodal precession against analytical theory.
"""

import numpy as np

from src.constants import MU_EARTH, R_EARTH, J2_EARTH
from src.dynamics import OrbitalDynamics
from src.propagation import propagate_orbit


def get_raan(solution):
    """
    Calculate longitude of ascending node (radians)
    throughout an orbital trajectory.
    """
    positions = solution.y[:3].T
    velocities = solution.y[3:].T

    # Orbital angular momentum vectors
    h = np.cross(positions, velocities)

    # Ascending-node vectors: N = z_hat x h
    z_hat = np.array([0.0, 0.0, 1.0])
    nodes = np.cross(z_hat, h)

    # Unwrap to avoid jumps from +pi to -pi
    raan = np.unwrap(np.arctan2(nodes[:, 1], nodes[:, 0]))

    return raan


def test_j2_precession():
    altitude = 500_000.0
    inclination = np.radians(55.0)

    radius = R_EARTH + altitude
    speed = np.sqrt(MU_EARTH / radius)

    initial_state = np.array(
        [
            radius,
            0.0,
            0.0,
            0.0,
            speed * np.cos(inclination),
            speed * np.sin(inclination),
        ]
    )

    duration = 48 * 3600
    times = np.linspace(0, duration, 1500)

    # Simulate orbit with J2
    solution_j2 = propagate_orbit(
        initial_state,
        (0, duration),
        dynamics=OrbitalDynamics(include_j2=True),
        t_eval=times,
    )

    # Simulate without J2
    solution_two_body = propagate_orbit(
        initial_state,
        (0, duration),
        dynamics=OrbitalDynamics(include_j2=False),
        t_eval=times,
    )

    # Calculate ascending-node angles
    raan_j2 = get_raan(solution_j2)
    raan_two_body = get_raan(solution_two_body)

    # Estimate average precession rates using linear fits
    slope_j2 = np.polyfit(times, raan_j2, 1)[0]
    slope_two_body = np.polyfit(times, raan_two_body, 1)[0]

    # Convert from rad/s to degrees/day
    conversion = 180 / np.pi * 86400

    measured_rate = slope_j2 * conversion
    two_body_rate = slope_two_body * conversion

    # Analytical J2 precession (e = 0)
    mean_motion = np.sqrt(MU_EARTH / radius**3)

    predicted_rate = (
        -1.5
        * J2_EARTH
        * mean_motion
        * (R_EARTH / radius) ** 2
        * np.cos(inclination)
        * conversion
    )

    print("\nJ2 Nodal Precession")
    print(f"Predicted rate: {predicted_rate:.5f} deg/day")
    print(f"Measured rate:  {measured_rate:.5f} deg/day")
    print(f"Two-body rate:  {two_body_rate:.5f} deg/day")

    error = abs(measured_rate - predicted_rate)
    print(f"Difference:     {error:.5f} deg/day")

    # Allow for higher-order and periodic J2 effects
    assert error < 0.1

    # A spherical gravitational field does not
    # produce nodal precession.
    assert abs(two_body_rate) < 1e-4
