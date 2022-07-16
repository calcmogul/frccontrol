import numpy as np
from scipy.linalg import expm


def discretize_ab(A, B, T):
    """Discretizes the given continuous A and B matrices.

    Keyword arguments:
    A -- continuous system matrix
    B -- continuous input matrix
    T -- discretization timestep

    Returns:
    discrete system matrix, discrete input matrix
    """
    states = A.shape[0]
    inputs = B.shape[1]

    M = expm(
        np.block([[A, B], [np.zeros((inputs, states)), np.zeros((inputs, states))]]) * T
    )
    return M[:states, :states], M[:states, states:]
