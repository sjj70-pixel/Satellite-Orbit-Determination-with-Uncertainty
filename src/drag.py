"""Optional atmospheric drag for a satellite truth simulation.

The original OrbitalDynamics class remains unchanged. To simulate drag,
wrap it with DragAugmentedDynamics and pass that wrapper to simulate_tracking.
Keep the batch estimator and EKF on their usual no-drag OrbitalDynamics
when testing sensitivity to a missing physical force.

All quantities use SI units and ECI Cartesian coordinates.
The atmosphere model is a simplified exponential profile, not a model of
solar activity, space weather, thermospheric winds, or satellite attitude.
"""

from dataclasses import dataclass, field

import numpy as np

from .constants import OMEGA_EARTH, R_EARTH, WGS84_FLATTENING
from .dynamics import OrbitalDynamics


def geodetic_altitude_m(position_eci):
    """Approximate altitude above the WGS84 ellipsoid from an ECI position.

    Because the WGS84 ellipsoid rotates about the ECI z-axis, geodetic
    altitude depends only on the distance from that axis and on z. We use
    Bowring's formula, which is accurate at ordinary low-Earth-orbit heights.
    """
    position = np.asarray(position_eci, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_eci must be a finite 3-vector")

    x, y, z = position
    a = R_EARTH  # WGS84 equatorial radius (m)
    b = a * (1.0 - WGS84_FLATTENING)
    e2 = 1.0 - (b / a) ** 2  # First eccentricity squared
    ep2 = (a / b) ** 2 - 1.0  # Second eccentricity squared
    p = np.hypot(x, y)

    if p < 1e-9:  # On the rotation axis (the poles)
        return float(abs(z) - b)

    theta = np.arctan2(z * a, p * b)
    latitude = np.arctan2(
        z + ep2 * b * np.sin(theta) ** 3,
        p - e2 * a * np.cos(theta) ** 3,
    )
    n = a / np.sqrt(1.0 - e2 * np.sin(latitude) ** 2)
    return float(p / np.cos(latitude) - n)


@dataclass(frozen=True)
class ExponentialAtmosphere:
    """A fixed exponential density profile, intended for roughly 400-600 km.

    rho(h) = rho_ref * exp(-(h - h_ref)/H).

    The default density and scale height are illustrative LEO values.
    In reality density depends strongly on solar and geomagnetic activity.
    """

    reference_altitude_m: float = 500_000.0
    reference_density_kg_m3: float = 7.0e-13
    scale_height_m: float = 65_000.0

    def __post_init__(self):
        if not np.isfinite(self.reference_altitude_m) or self.reference_altitude_m < 0:
            raise ValueError("reference_altitude_m must be finite and nonnegative")
        if (
            not np.isfinite(self.reference_density_kg_m3)
            or self.reference_density_kg_m3 <= 0
        ):
            raise ValueError("reference_density_kg_m3 must be finite and positive")
        if not np.isfinite(self.scale_height_m) or self.scale_height_m <= 0:
            raise ValueError("scale_height_m must be finite and positive")

    def density(self, altitude_m):
        """Return atmospheric mass density (kg/m^3) at geodetic altitude (m)."""
        if not np.isfinite(altitude_m):
            raise ValueError("altitude_m must be finite")
        exponent = -(altitude_m - self.reference_altitude_m) / self.scale_height_m
        return float(self.reference_density_kg_m3 * np.exp(exponent))


@dataclass(frozen=True)
class AtmosphericDrag:
    """Acceleration from quadratic drag relative to the co-rotating air.

    a_drag = -(1/2) * rho * C_D * (A/m) * |v_rel| * v_rel,
    v_rel = v_sat - (omega_Earth cross r_sat).
    """

    drag_coefficient: float = 2.2
    cross_section_m2: float = 0.2
    mass_kg: float = 20.0
    atmosphere: ExponentialAtmosphere = field(default_factory=ExponentialAtmosphere)

    def __post_init__(self):
        if not np.isfinite(self.drag_coefficient) or self.drag_coefficient <= 0:
            raise ValueError("drag_coefficient must be finite and positive")
        if not np.isfinite(self.cross_section_m2) or self.cross_section_m2 < 0:
            raise ValueError("cross_section_m2 must be finite and nonnegative")
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError("mass_kg must be finite and positive")

    def acceleration(self, time_s, state):
        """Return ECI drag acceleration (m/s^2) for a 6D satellite state."""
        state = np.asarray(state, dtype=float)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite array of shape (6,)")

        position = state[:3]
        atmosphere_velocity = np.cross(np.array([0.0, 0.0, OMEGA_EARTH]), position)
        relative_velocity = state[3:] - atmosphere_velocity
        speed = np.linalg.norm(relative_velocity)
        density = self.atmosphere.density(geodetic_altitude_m(position))
        return (
            -0.5
            * density
            * self.drag_coefficient
            * self.cross_section_m2
            / self.mass_kg
            * speed
            * relative_velocity
        )


@dataclass
class DragAugmentedDynamics:
    """Combine ordinary gravity and optional drag without changing src/dynamics.py.

    This callable can replace OrbitalDynamics as the dynamics argument to
    simulate_tracking() or propagate_orbit(). It is intended for truth
    propagation; estimators can continue using OrbitalDynamics(include_j2=True).
    """

    base_dynamics: OrbitalDynamics = field(
        default_factory=lambda: OrbitalDynamics(include_j2=True)
    )
    drag_model: AtmosphericDrag = field(default_factory=AtmosphericDrag)

    def __call__(self, time_s, state):
        derivative = np.asarray(self.base_dynamics(time_s, state), dtype=float).copy()
        derivative[3:] += self.drag_model.acceleration(time_s, state)
        return derivative
