import control as cnt
import numpy as np
import scipy as sp


def dlqr(sys, Q, R):
    """Solves for the optimal discrete-time LQR controller.

    x(n+1) = A * x(n) + B * u(n)
    J = sum(0, inf, x.T * Q * x + u.T * R * u)

    Keyword arguments:
    A -- numpy.array(states x states), The A matrix.
    B -- numpy.array(inputs x states), The B matrix.
    Q -- numpy.array(states x states), The state cost matrix.
    R -- numpy.array(inputs x inputs), The control effort cost matrix.

    Returns:
    numpy.array(states x inputs), K
    """
    m = sys.A.shape[0]

    controllability_rank = np.linalg.matrix_rank(cnt.ctrb(sys.A, sys.B))
    if controllability_rank != m:
        print(
            "Warning: Controllability of %d != %d, uncontrollable state"
            % (controllability_rank, m)
        )

    # P = A.T * P * A - (A.T * P * B) * np.linalg.inv(R + B.T * P * B) *
    #     (B.T * P.T * A) + Q
    P = sp.linalg.solve_discrete_are(a=sys.A, b=sys.B, q=Q, r=R)

    F = np.linalg.inv(R + sys.B.T * P * sys.B) * sys.B.T * P * sys.A
    return F
