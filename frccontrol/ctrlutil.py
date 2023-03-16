"""Control system utility functions."""

import numpy as np


def make_cost_matrix(elems):
    """Creates a cost matrix from the given vector for use with LQR.

    The cost matrix is constructed using Bryson's rule. The inverse square
    of each element in the input is taken and placed on the cost matrix
    diagonal.

    Keyword arguments:
    elems -- a vector. For a Q matrix, its elements are the maximum allowed
             excursions of the states from the reference. For an R matrix,
             its elements are the maximum allowed excursions of the control
             inputs from no actuation.

    Returns:
    State excursion or control effort cost matrix
    """
    return np.diag([0.0 if elem == float("inf") else 1.0 / elem**2 for elem in elems])


def make_cov_matrix(elems):
    """Creates a covariance matrix from the given vector for use with Kalman
    filters.

    Each element is squared and placed on the covariance matrix diagonal.

    Keyword arguments:
    elems -- a vector. For a Q matrix, its elements are the standard
             deviations of each state from how the model behaves. For an R
             matrix, its elements are the standard deviations for each
             output measurement.

    Returns:
    Process noise or measurement noise covariance matrix
    """
    return np.diag(np.square(elems))


def ctrb(A, B):
    """Returns the controllability matrix for A and B

    Keyword arguments:
    A -- system matrix
    B -- input matrix
    """
    return np.hstack(
        [B] + [np.linalg.matrix_power(A, i) @ B for i in range(1, A.shape[0])]
    )


def obsv(A, C):
    """Returns the observability matrix for A and C

    Keyword arguments:
    A -- system matrix
    C -- output matrix
    """
    return np.vstack(
        [C] + [C @ np.linalg.matrix_power(A, i) for i in range(1, A.shape[0])]
    )
