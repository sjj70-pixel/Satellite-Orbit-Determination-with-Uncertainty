import numpy as np

from src.constants import MU_EARTH, R_EARTH
from src.propagation import propagate_orbit


def circular_orbit_initial_conditions():
    radius = R_EARTH + 500_000
    speed = np.sqrt(MU_EARTH / radius)
    period = 2 * np.pi * np.sqrt(radius**3 / MU_EARTH)

    initial_state = np.array([radius, 0.0, 0.0, 0.0, speed, 0.0])

    return initial_state, period


def test_circular_orbit_one_period():
    initial_state, period = circular_orbit_initial_conditions()

    solution = propagate_orbit(
        initial_state,
        t_span=(0, period),
    )

    final_state = solution.y[:, -1]

    position_error = np.linalg.norm(final_state[:3] - initial_state[:3])

    velocity_error = np.linalg.norm(final_state[3:] - initial_state[3:])

    assert position_error < 0.1  # meters
    assert velocity_error < 1e-4  # m/s


def test_energy_conservation():
    initial_state, period = circular_orbit_initial_conditions()

    solution = propagate_orbit(
        initial_state,
        t_span=(0, period),
        t_eval=np.linspace(0, period, 500),
    )

    positions = solution.y[:3, :]
    velocities = solution.y[3:, :]

    radii = np.linalg.norm(positions, axis=0)
    speeds = np.linalg.norm(velocities, axis=0)

    energy = 0.5 * speeds**2 - MU_EARTH / radii

    relative_error = np.abs((energy - energy[0]) / energy[0])

    assert np.max(relative_error) < 1e-8
