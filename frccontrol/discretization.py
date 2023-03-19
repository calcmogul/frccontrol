"""Functions for discretizing state-space systems."""

import numpy as np
from scipy.linalg import expm


def discretize_a(A, dt):
    """Discretizes the given continuous A matrix.

    Keyword arguments:
    A -- continuous system matrix
    dt -- discretization timestep

    Returns:
    discrete system matrix
    """
    # ϕ = eᴬᵀ
    return expm(A * dt)


def discretize_ab(A, B, dt):
    """Discretizes the given continuous A and B matrices.

    Keyword arguments:
    A -- continuous system matrix
    B -- continuous input matrix
    dt -- discretization timestep

    Returns:
    discrete system matrix, discrete input matrix
    """
    states = A.shape[0]
    inputs = B.shape[1]

    # M = [A  B]
    #     [0  0]
    M = expm(
        np.block([[A, B], [np.zeros((inputs, states)), np.zeros((inputs, inputs))]])
        * dt
    )

    # ϕ = eᴹᵀ = [A_d  B_d]
    #           [ 0    I ]
    return M[:states, :states], M[:states, states:]


def discretize_aq(A, Q, dt):
    """
    Discretizes the given continuous A and Q matrices.

    Keyword arguments:
    A -- Continuous system matrix.
    Q -- Continuous process noise covariance matrix.
    dt -- Discretization timestep.

    Returns:
    discrete system matrix, discrete process noise covariance matrix
    """
    states = A.shape[0]

    # Make continuous Q symmetric if it isn't already
    contQ = (Q + Q.T) / 2.0

    # M = [−A  Q ]
    #     [ 0  Aᵀ]
    M = np.block([[-A, contQ], [np.zeros((states, states)), A.T]])

    # ϕ = eᴹᵀ = [−A_d  A_d⁻¹Q_d]
    #           [ 0      A_dᵀ  ]
    phi = expm(M * dt)

    # ϕ₁₂ = A_d⁻¹Q_d
    phi12 = phi[:states, states:]

    # ϕ₂₂ = A_dᵀ
    phi22 = phi[states:, states:]

    discA = phi22.T
    discQ = discA @ phi12

    # Make discrete Q symmetric if it isn't already
    return discA, (discQ + discQ.T) / 2.0


def discretize_r(R, dt):
    """
    Returns a discretized version of the provided continuous measurement noise
    covariance matrix.

    Keyword arguments:
    R -- Continuous measurement noise covariance matrix.
    dt -- Discretization timestep.

    Returns:
    discrete measurement noise covariance matrix
    """
    # R_d = 1/T R
    return R / dt
