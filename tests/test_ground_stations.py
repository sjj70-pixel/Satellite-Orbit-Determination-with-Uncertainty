"""Tests for rotating, elliptical-Earth ground-station geometry."""

import numpy as np
import pytest

from src.constants import OMEGA_EARTH, R_EARTH, WGS84_FLATTENING
from src.ground_stations import GroundStation


def test_station_position_at_initial_time():
    station = GroundStation("Equator", latitude_deg=0.0, longitude_deg=0.0)
    np.testing.assert_allclose(
        station.position_eci(0.0),
        [R_EARTH, 0.0, 0.0],
        atol=1e-6,
    )


def test_station_rotates_one_quarter_turn():
    station = GroundStation("Equator", latitude_deg=0.0, longitude_deg=0.0)
    quarter_rotation = np.pi / (2.0 * OMEGA_EARTH)
    np.testing.assert_allclose(
        station.position_eci(quarter_rotation),
        [0.0, R_EARTH, 0.0],
        atol=1e-6,
    )


def test_station_velocity():
    station = GroundStation("Equator", latitude_deg=0.0, longitude_deg=0.0)
    np.testing.assert_allclose(
        station.velocity_eci(0.0),
        [0.0, OMEGA_EARTH * R_EARTH, 0.0],
        atol=1e-10,
    )


def test_overhead_and_opposite_side_visibility():
    station = GroundStation("Equator", latitude_deg=0.0, longitude_deg=0.0)
    overhead = np.array([R_EARTH + 500_000.0, 0.0, 0.0])
    opposite = np.array([-R_EARTH - 500_000.0, 0.0, 0.0])

    assert np.isclose(station.elevation_angle(overhead, 0.0), np.pi / 2)
    assert np.isclose(station.elevation_angle(opposite, 0.0), -np.pi / 2)
    assert station.is_visible(overhead, 0.0)
    assert not station.is_visible(opposite, 0.0)


def test_invalid_coordinates():
    with pytest.raises(ValueError):
        GroundStation("Invalid", latitude_deg=95.0, longitude_deg=0.0)


def test_polar_radius():
    """
    Verify that a station at the North Pole lies
    at the WGS84 polar radius.
    """
    station = GroundStation(name="North Pole", latitude_deg=90.0, longitude_deg=0.0)

    polar_radius = R_EARTH * (1 - WGS84_FLATTENING)

    expected = np.array([0.0, 0.0, polar_radius])

    np.testing.assert_allclose(station.position_eci(0.0), expected, atol=1e-6)


def test_station_altitude():
    """
    Verify station altitude above the ellipsoid
    at the equator.
    """
    station = GroundStation(
        name="Elevated Station", latitude_deg=0.0, longitude_deg=0.0, altitude_m=100.0
    )

    expected = np.array([R_EARTH + 100.0, 0.0, 0.0])

    np.testing.assert_allclose(station.position_eci(0.0), expected, atol=1e-6)


def test_geodetic_local_vertical():
    """
    Verify that the local vertical is normal to
    the ellipsoid rather than radial from Earth's
    center.
    """
    station = GroundStation(
        name="Midlatitude Station", latitude_deg=45.0, longitude_deg=0.0
    )

    position = station.position_eci(0.0)
    local_up = station.local_up_eci(0.0)

    # WGS84 equatorial and polar radii
    a = R_EARTH
    b = a * (1 - WGS84_FLATTENING)

    # Surface normal of the ellipsoid
    expected_normal = np.array(
        [position[0] / a**2, position[1] / a**2, position[2] / b**2]
    )

    expected_normal /= np.linalg.norm(expected_normal)

    np.testing.assert_allclose(local_up, expected_normal, atol=1e-12)

    # Place satellite 500 km directly along
    # the local vertical.
    satellite_position = position + 500_000 * local_up

    elevation = station.elevation_angle(satellite_position, t=0.0)

    assert np.isclose(elevation, np.pi / 2)
