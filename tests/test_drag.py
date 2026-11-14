"""Tests of the simplified atmospheric density and satellite drag model."""

import numpy as np
import pytest

from src.constants import MU_EARTH, OMEGA_EARTH, R_EARTH, WGS84_FLATTENING
from src.drag import (
    AtmosphericDrag,
    DragAugmentedDynamics,
    ExponentialAtmosphere,
    geodetic_altitude_m,
)
from src.dynamics import OrbitalDynamics
from src.propagation import propagate_orbit


def circular_orbit():
    radius = R_EARTH + 500_000.0
    return np.array([radius, 0.0, 0.0, 0.0, np.sqrt(MU_EARTH / radius), 0.0])


def test_wgs84_altitude_at_equator_pole_and_midlatitude():
    altitude = 500_000.0
    a = R_EARTH
    b = a * (1.0 - WGS84_FLATTENING)
    np.testing.assert_allclose(
        geodetic_altitude_m([a + altitude, 0.0, 0.0]), altitude, atol=1e-6
    )
    np.testing.assert_allclose(
        geodetic_altitude_m([0.0, 0.0, b + altitude]), altitude, atol=1e-6
    )

    latitude = np.deg2rad(55.0)
    e2 = 1.0 - (b / a) ** 2
    n = a / np.sqrt(1.0 - e2 * np.sin(latitude) ** 2)
    position = [
        (n + altitude) * np.cos(latitude),
        0.0,
        (n * (1.0 - e2) + altitude) * np.sin(latitude),
    ]
    assert abs(geodetic_altitude_m(position) - altitude) < 0.1


def test_exponential_density_scales_by_e():
    atmosphere = ExponentialAtmosphere()
    altitude = atmosphere.reference_altitude_m
    scale_height = atmosphere.scale_height_m
    assert atmosphere.density(altitude) == pytest.approx(
        atmosphere.reference_density_kg_m3
    )
    assert atmosphere.density(altitude + scale_height) == pytest.approx(
        atmosphere.reference_density_kg_m3 / np.e
    )
    assert atmosphere.density(altitude - scale_height) == pytest.approx(
        atmosphere.reference_density_kg_m3 * np.e
    )


def test_drag_is_opposite_relative_velocity_and_has_correct_magnitude():
    state = circular_orbit()
    drag = AtmosphericDrag()
    air_velocity = np.cross([0.0, 0.0, OMEGA_EARTH], state[:3])
    v_relative = state[3:] - air_velocity
    acceleration = drag.acceleration(0.0, state)
    density = drag.atmosphere.density(500_000.0)
    expected_magnitude = (
        0.5
        * density
        * drag.drag_coefficient
        * drag.cross_section_m2
        / drag.mass_kg
        * np.linalg.norm(v_relative) ** 2
    )

    assert np.dot(acceleration, v_relative) < 0.0
    assert np.linalg.norm(acceleration) == pytest.approx(expected_magnitude)
    assert acceleration[1] < 0.0


def test_drag_vanishes_if_satellite_comoves_with_atmosphere():
    state = circular_orbit()
    state[3:] = np.cross([0.0, 0.0, OMEGA_EARTH], state[:3])
    np.testing.assert_array_equal(
        AtmosphericDrag().acceleration(0.0, state), np.zeros(3)
    )


def test_drag_strength_scales_with_area_over_mass():
    state = circular_orbit()
    default_acc = AtmosphericDrag().acceleration(0.0, state)
    double_area_acc = AtmosphericDrag(cross_section_m2=0.4).acceleration(0.0, state)
    double_mass_acc = AtmosphericDrag(mass_kg=40.0).acceleration(0.0, state)

    np.testing.assert_allclose(double_area_acc, 2.0 * default_acc)
    np.testing.assert_allclose(double_mass_acc, 0.5 * default_acc)


def test_augmented_dynamics_adds_only_drag_acceleration():
    state = circular_orbit()
    base = OrbitalDynamics(include_j2=True)
    drag = AtmosphericDrag()
    augmented = DragAugmentedDynamics(base_dynamics=base, drag_model=drag)

    base_derivative = base(0.0, state)
    augmented_derivative = augmented(0.0, state)
    np.testing.assert_allclose(
        augmented_derivative[:3], base_derivative[:3], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        augmented_derivative[3:] - base_derivative[3:],
        drag.acceleration(0.0, state),
        rtol=1e-9,
        atol=1e-15,
    )


def test_drag_reduces_two_body_specific_orbital_energy():
    state0 = circular_orbit()
    dynamics = DragAugmentedDynamics(base_dynamics=OrbitalDynamics(include_j2=False))
    trajectory = propagate_orbit(
        initial_state=state0,
        t_span=(0.0, 3600.0),
        dynamics=dynamics,
        t_eval=[3600.0],
    )
    state1 = trajectory.y[:, -1]

    def energy(state):
        return np.dot(state[3:], state[3:]) / 2.0 - MU_EARTH / np.linalg.norm(state[:3])

    assert energy(state1) < energy(state0) - 1.0


def test_invalid_drag_parameters_are_rejected():
    with pytest.raises(ValueError):
        AtmosphericDrag(mass_kg=0.0)
    with pytest.raises(ValueError):
        AtmosphericDrag(drag_coefficient=-1.0)
    with pytest.raises(ValueError):
        AtmosphericDrag(cross_section_m2=-0.1)
    with pytest.raises(ValueError):
        ExponentialAtmosphere(scale_height_m=0.0)
    with pytest.raises(ValueError):
        ExponentialAtmosphere(reference_density_kg_m3=-1.0)
    with pytest.raises(ValueError):
        geodetic_altitude_m([1.0, 2.0])
