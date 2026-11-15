"""Idealized Earth-fixed ground stations expressed in an inertial frame.

The Earth is modeled as an ellipsoid rotating at a constant angular rate about
the ECI z-axis. At simulation time t=0, zero longitude points along +x.

Lengths are meters, times are seconds, and angles returned by this module
are radians. Latitude and longitude constructor arguments are in degrees.
"""

from dataclasses import dataclass

import numpy as np

from .constants import OMEGA_EARTH, R_EARTH, WGS84_FLATTENING


@dataclass(frozen=True)
class GroundStation:
    """An idealized tracking station on the surface of the WGS84 ellipsoid.

    Parameters
    ----------
    name : str
        Station identifier.
    latitude_deg : float
        Geocentric latitude in degrees (-90 to 90).
    longitude_deg : float
        East-positive longitude in degrees (-180 to 180).
    min_elevation_deg : float
        Minimum observable satellite elevation in degrees (0 to 90).

    The ECI x-axis is aligned with longitude 0 at t=0. A real dated
    Earth-fixed-to-inertial transformation would need an epoch and Earth
    orientation model; geodetic latitude would also require an ellipsoid.
    """

    name: str
    latitude_deg: float
    longitude_deg: float
    min_elevation_deg: float = 10.0
    altitude_m: float = 0.0

    def __post_init__(self):
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("latitude_deg must be between -90 and 90")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError("longitude_deg must be between -180 and 180")
        if not 0.0 <= self.min_elevation_deg <= 90.0:
            raise ValueError("min_elevation_deg must be between 0 and 90")
        if not np.isfinite(self.altitude_m):
            raise ValueError("altitude_m must be finite")

    def position_eci(self, t):
        """
        Calculate ground-station position in ECI coordinates.

        Uses the WGS84 reference ellipsoid and assumes
        constant Earth rotation about the z-axis.

        Parameters
        ----------
        t : float
            Time since simulation epoch (seconds).

        Returns
        -------
        ndarray, shape (3,)
            Station position in ECI coordinates (meters).
        """
        latitude = np.deg2rad(self.latitude_deg)

        longitude = np.deg2rad(self.longitude_deg) + OMEGA_EARTH * t

        # First eccentricity squared of the WGS84 ellipsoid
        f = WGS84_FLATTENING
        e_squared = 2 * f - f**2

        sin_lat = np.sin(latitude)
        cos_lat = np.cos(latitude)

        # Radius of curvature in the prime vertical
        N = R_EARTH / np.sqrt(1 - e_squared * sin_lat**2)

        h = self.altitude_m

        x = (N + h) * cos_lat * np.cos(longitude)
        y = (N + h) * cos_lat * np.sin(longitude)
        z = (N * (1 - e_squared) + h) * sin_lat

        return np.array([x, y, z])

    def local_up_eci(self, t):
        """
        Return the outward unit normal to the WGS84
        ellipsoid at the station's location, in ECI.
        """
        latitude = np.deg2rad(self.latitude_deg)

        longitude = np.deg2rad(self.longitude_deg) + OMEGA_EARTH * t

        return np.array(
            [
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            ]
        )

    def velocity_eci(self, t):
        """Return station velocity in ECI coordinates (m/s)."""
        x, y, _ = self.position_eci(t)
        return OMEGA_EARTH * np.array([-y, x, 0.0])

    def elevation_angle(self, satellite_position, t):
        """Return the satellite's elevation above the local horizon (rad).

        A value of pi/2 means directly overhead; zero means on the
        geometric horizon; negative values mean below the horizon.
        """
        satellite_position = np.asarray(satellite_position, dtype=float)
        if satellite_position.shape != (3,):
            raise ValueError("satellite_position must have shape (3,)")

        station_position = self.position_eci(t)
        line_of_sight = satellite_position - station_position
        distance = np.linalg.norm(line_of_sight)
        if distance == 0.0:
            raise ValueError("satellite and station cannot have identical positions")

        local_up = self.local_up_eci(t)
        sine_elevation = np.dot(line_of_sight, local_up) / distance

        # Clamp to handle floating-point rounding near +/- 90 degrees.
        return np.arcsin(np.clip(sine_elevation, -1.0, 1.0))

    def is_visible(self, satellite_position, t):
        """Return True if the satellite meets the minimum elevation mask."""
        return bool(
            self.elevation_angle(satellite_position, t)
            >= np.deg2rad(self.min_elevation_deg)
        )
