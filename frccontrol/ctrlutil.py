"""Control system utility functions."""

import numpy as np
import scipy as sp


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


def is_stabilizable(A, B):
    """Returns true if (A, B) is a stabilizable pair.

    (A, B) is stabilizable if and only if the uncontrollable eigenvalues of A,
    if any, have absolute values less than one, where an eigenvalue is
    uncontrollable if rank([λI - A, B]) < n where n is the number of states.

    Keyword arguments:
    A -- system matrix
    B -- input matrix
    """
    rows = A.shape[0]
    w, _ = sp.linalg.eig(A)

    for i in range(rows):
        if abs(w[i]) < 1:
            continue

        E = np.block([[w[i] * np.eye(rows) - A, B]])

        if np.linalg.matrix_rank(E) < rows:
            return False
    return True


def is_detectable(A, C):
    """Returns true if (A, C) is a detectable pair.

    (A, C) is detectable if and only if the unobservable eigenvalues of A, if
    any, have absolute values less than one, where an eigenvalue is unobservable
    if rank([λI - A; C]) < n where n is the number of states.

    Keyword arguments:
    A -- system matrix
    C -- output matrix
    """
    return is_stabilizable(A.T, C.T)


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
