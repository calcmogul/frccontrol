import numpy as np


def numerical_jacobian(rows, cols, f, x):
    """Returns numerical Jacobian with respect to x for f(x).

    Keyword arguments:
    rows -- Number of rows in result of f(x).
    cols -- Number of columns in result of f(x).
    f -- Vector-valued function from which to compute Jacobian.
    x -- Vector argument.
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
    """Returns numerical Jacobian with respect to x for f(x, u).

    rows -- Number of rows in result of f(x, u).
    cols -- Number of columns in result of f(x, u).
    f -- Vector-valued function from which to compute Jacobian.
    x -- State vector.
    u -- Input vector.
    """
    return numerical_jacobian(rows, cols, lambda x: f(x, u), x)


def numerical_jacobian_u(rows, cols, f, x, u):
    """Returns numerical Jacobian with respect to u for f(x, u).

    rows -- Number of rows in result of f(x, u).
    cols -- Number of columns in result of f(x, u).
    f -- Vector-valued function from which to compute Jacobian.
    x -- State vector.
    u -- Input vector.
    """
    return numerical_jacobian(rows, cols, lambda u: f(x, u), u)
