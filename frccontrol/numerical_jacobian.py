"""Functions for computing the numerical Jacobian of nonlinear functions."""

import numpy as np


def numerical_jacobian(rows, cols, f, x):
    """
    Returns numerical Jacobian with respect to x for f(x).

    Parameter ``rows``:
        Number of rows in result of f(x).
    Parameter ``cols``:
        Number of columns in result of f(x).
    Parameter ``f``:
        Vector-valued function from which to compute Jacobian.
    Parameter ``x``:
        Vector argument.
    """
    EPSILON = 1e-5
    result = np.zeros((rows, cols))

    for i in range(cols):
        dx_plus = np.copy(x)
        dx_plus[i, 0] += EPSILON
        dx_minus = np.copy(x)
        dx_minus[i, 0] -= EPSILON
        result[:, i : i + 1] = (f(dx_plus) - f(dx_minus)) / (EPSILON * 2.0)
    return result


def numerical_jacobian_x(rows, cols, f, x, u):
    """
    Returns numerical Jacobian with respect to x for f(x, u).

    Parameter ``rows``:
        Number of rows in result of f(x, u).
    Parameter ``cols``:
        Number of columns in result of f(x, u).
    Parameter ``f``:
        Vector-valued function from which to compute Jacobian.
    Parameter ``x``:
        State vector.
    Parameter ``u``:
        Input vector.
    """
    return numerical_jacobian(rows, cols, lambda x: f(x, u), x)


def numerical_jacobian_u(rows, cols, f, x, u):
    """
    Returns numerical Jacobian with respect to u for f(x, u).

    Parameter ``rows``:
        Number of rows in result of f(x, u).
    Parameter ``cols``:
        Number of columns in result of f(x, u).
    Parameter ``f``:
        Vector-valued function from which to compute Jacobian.
    Parameter ``x``:
        State vector.
    Parameter ``u``:
        Input vector.
    """
    return numerical_jacobian(rows, cols, lambda u: f(x, u), u)
