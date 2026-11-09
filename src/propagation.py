"""
Numerical propagation of satellite orbits.

Integrates the six-dimensional orbital state:
    [x, y, z, vx, vy, vz]

All quantities use SI units.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import OrbitalDynamics


def propagate_orbit(
    initial_state,
    t_span,
    dynamics=None,
    t_eval=None,
    rtol=1e-10,
    atol=None,
):
    """
    Numerically propagate a satellite's orbit.

    Parameters
    ----------
    initial_state : array_like, shape (6,)
        Initial position (m) and velocity (m/s).

    t_span : tuple
        (t_start, t_end) in seconds.

    dynamics : OrbitalDynamics, optional
        Gravitational model to use.
        Defaults to two-body gravity without J2.

    t_eval : array_like, optional
        Times at which to return the satellite state.
        If None, returns the solver's adaptive time points.

    rtol : float
        Relative integration tolerance.

    atol : array_like or float, optional
        Absolute integration tolerances.
        Defaults to 1 mm for position and
        1 micrometer/s for velocity.

    Returns
    -------
    OdeResult
        SciPy solution object containing:
            t : evaluation times
            y : satellite states, shape (6, N)
            success : whether integration succeeded
            message : solver status message
    """
    initial_state = np.asarray(initial_state, dtype=float)

    if initial_state.shape != (6,):
        raise ValueError("initial_state must have shape (6,)")

    if dynamics is None:
        dynamics = OrbitalDynamics(include_j2=False)

    if atol is None:
        atol = np.array([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6])

    solution = solve_ivp(
        fun=dynamics,
        t_span=t_span,
        y0=initial_state,
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    if not solution.success:
        raise RuntimeError(f"Orbital propagation failed: {solution.message}")

    return solution
