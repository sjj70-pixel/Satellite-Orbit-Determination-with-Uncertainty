"""
Orbital dynamics for Earth-orbiting satellites.

Supports:
    - Newtonian two-body gravitational acceleration
    - Earth's J2 gravitational perturbation
    - Six-dimensional Cartesian state derivatives

Coordinate system:
    Earth-centered inertial (ECI), with the z-axis aligned
    with Earth's rotation axis.

All quantities use SI units.
"""

import numpy as np

from .constants import MU_EARTH, R_EARTH, J2_EARTH


def two_body_acceleration(position):
    """
    Calculate Newtonian gravitational acceleration.

    Parameters
    ----------
    position : array_like, shape (3,)
        Satellite position [x, y, z] in meters.

    Returns
    -------
    ndarray, shape (3,)
        Gravitational acceleration in m/s^2.
    """
    r = np.asarray(position, dtype=float)
    r_norm = np.linalg.norm(r)

    if r_norm == 0:
        raise ValueError("Position cannot be the zero vector.")

    return -MU_EARTH * r / r_norm**3


def j2_acceleration(position):
    """
    Calculate the J2 gravitational perturbation.

    Parameters
    ----------
    position : array_like, shape (3,)
        Satellite position [x, y, z] in meters.

    Returns
    -------
    ndarray, shape (3,)
        J2 acceleration correction in m/s^2.
    """
    r = np.asarray(position, dtype=float)
    x, y, z = r

    r_norm = np.linalg.norm(r)

    if r_norm == 0:
        raise ValueError("Position cannot be the zero vector.")

    z_ratio_sq = (z / r_norm) ** 2

    factor = 1.5 * J2_EARTH * MU_EARTH * R_EARTH**2 / r_norm**5

    return factor * np.array(
        [x * (5 * z_ratio_sq - 1), y * (5 * z_ratio_sq - 1), z * (5 * z_ratio_sq - 3)]
    )


class OrbitalDynamics:
    """
    Orbital dynamics model with optional J2 perturbation.

    Parameters
    ----------
    include_j2 : bool
        Whether to include Earth's J2 perturbation.
    """

    def __init__(self, include_j2=False):
        self.include_j2 = include_j2

    def acceleration(self, position):
        """
        Calculate total gravitational acceleration.
        """
        a = two_body_acceleration(position)

        if self.include_j2:
            a += j2_acceleration(position)

        return a

    def __call__(self, t, state):
        """
        Calculate the time derivative of the orbital state.

        Parameters
        ----------
        t : float
            Simulation time in seconds.
        state : array_like, shape (6,)
            [x, y, z, vx, vy, vz]

        Returns
        -------
        ndarray, shape (6,)
            [vx, vy, vz, ax, ay, az]
        """
        position = state[:3]
        velocity = state[3:]

        acceleration = self.acceleration(position)

        return np.concatenate((velocity, acceleration))
