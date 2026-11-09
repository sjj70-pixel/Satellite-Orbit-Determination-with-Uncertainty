"""
Physical constants for satellite orbit determination.

All quantities use SI units:
    Distance: meters (m)
    Time: seconds (s)
    Mass: kilograms (kg)
    Angles: radians (rad)
"""

# Earth's gravitational parameter (m^3/s^2)
MU_EARTH = 3.986004418e14

# Earth's equatorial radius (m), WGS84
R_EARTH = 6_378_137.0

# Earth's second zonal harmonic coefficient (dimensionless)
J2_EARTH = 1.08262668e-3

# Earth's sidereal rotation rate (rad/s)
OMEGA_EARTH = 7.2921150e-5
