"""Functions for integrating nonlinear system dynamics."""

import math
import numpy as np


def rk4(f, x, u, dt):
    """Performs 4th order Runge-Kutta integration of dx/dt = f(x, u) for dt.

    Keyword arguments:
    f -- vector function to integrate
    x -- vector of states
    u -- vector of inputs (constant for dt)
    dt -- time for which to integrate
    """
    half_dt = dt * 0.5
    k1 = f(x, u)
    k2 = f(x + half_dt * k1, u)
    k3 = f(x + half_dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rkdp(f, x, u, dt, max_error=1e-6):
    """Performs adaptive Dormand-Prince integration of dx/dt = f(x, u) for dt.

    Keyword arguments:
    f -- vector function to integrate
    x -- vector of states
    u -- vector of inputs (constant for dt)
    dt -- time for which to integrate
    """
    A = np.empty((6, 6))

    A[0, :1] = np.array([1.0 / 5.0])
    A[1, :2] = np.array([3.0 / 40.0, 9.0 / 40.0])
    A[2, :3] = np.array([44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0])
    A[3, :4] = np.array(
        [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0]
    )
    A[4, :5] = np.array(
        [
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        ]
    )
    A[5, :6] = np.array(
        [
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ]
    )

    b1 = np.array(
        [
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
            0.0,
        ]
    )
    b2 = np.array(
        [
            5179.0 / 57600.0,
            0.0,
            7571.0 / 16695.0,
            393.0 / 640.0,
            -92097.0 / 339200.0,
            187.0 / 2100.0,
            1.0 / 40.0,
        ]
    )

    truncation_error = float("inf")

    dt_elapsed = 0.0
    h = dt

    # Loop until we've gotten to our desired dt
    while dt_elapsed < dt:
        while truncation_error > max_error:
            # Only allow us to advance up to the dt remaining
            h = min(h, dt - dt_elapsed)

            k1 = f(x, u)
            k2 = f(x + h * (A[0, 0] * k1), u)
            k3 = f(x + h * (A[1, 0] * k1 + A[1, 1] * k2), u)
            k4 = f(x + h * (A[2, 0] * k1 + A[2, 1] * k2 + A[2, 2] * k3), u)
            k5 = f(
                x + h * (A[3, 0] * k1 + A[3, 1] * k2 + A[3, 2] * k3 + A[3, 3] * k4), u
            )
            k6 = f(
                x
                + h
                * (
                    A[4, 0] * k1
                    + A[4, 1] * k2
                    + A[4, 2] * k3
                    + A[4, 3] * k4
                    + A[4, 4] * k5
                ),
                u,
            )

            # Since the final row of A and the array b1 have the same coefficients
            # and k7 has no effect on newX, we can reuse the calculation.
            new_x = x + h * (
                A[5, 0] * k1
                + A[5, 1] * k2
                + A[5, 2] * k3
                + A[5, 3] * k4
                + A[5, 4] * k5
                + A[5, 5] * k6
            )
            k7 = f(new_x, u)

            truncation_error = np.linalg.norm(
                h
                * (
                    (b1[0] - b2[0]) * k1
                    + (b1[1] - b2[1]) * k2
                    + (b1[2] - b2[2]) * k3
                    + (b1[3] - b2[3]) * k4
                    + (b1[4] - b2[4]) * k5
                    + (b1[5] - b2[5]) * k6
                    + (b1[6] - b2[6]) * k7
                )
            )

            if truncation_error == 0.0:
                h = dt - dt_elapsed
            else:
                h *= 0.9 * math.pow(max_error / truncation_error, 1.0 / 5.0)

        dt_elapsed += h
        x = new_x

    return x
