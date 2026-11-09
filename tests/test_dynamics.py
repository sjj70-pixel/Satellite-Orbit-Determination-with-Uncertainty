import numpy as np

from src.constants import MU_EARTH, R_EARTH
from src.dynamics import two_body_acceleration, j2_acceleration, OrbitalDynamics


def test_two_body_acceleration():
    r = R_EARTH + 500_000
    position = np.array([r, 0.0, 0.0])

    acceleration = two_body_acceleration(position)

    expected = np.array([-MU_EARTH / r**2, 0.0, 0.0])

    np.testing.assert_allclose(acceleration, expected)


def test_j2_direction():
    position = np.array([R_EARTH + 500_000, 0.0, 0.0])

    acceleration = j2_acceleration(position)

    assert acceleration[0] < 0
    assert acceleration[1] == 0
    assert acceleration[2] == 0


def test_dynamics_state_derivative():
    position = np.array([R_EARTH + 500_000, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])

    state = np.concatenate((position, velocity))

    dynamics = OrbitalDynamics(include_j2=False)
    derivative = dynamics(0.0, state)

    np.testing.assert_allclose(derivative[:3], velocity)
    np.testing.assert_allclose(derivative[3:], two_body_acceleration(position))
