import control as ct
import numpy as np
import scipy as sp


def kalmd(sys, Q, R):
    """Solves for the steady-state kalman gain matrix.

    Keyword arguments:
    sys -- discrete state-space model
    Q -- process noise covariance matrix
    R -- measurement noise covariance matrix

    Returns:
    K -- numpy.array(states x outputs), Kalman gain matrix.
    """
    if np.linalg.matrix_rank(ct.obsv(sys.A, sys.C)) < sys.A.shape[0]:
        print(f"Warning: The system is unobservable\n\nA = {sys.A}\nC = {sys.C}\n")

    P = sp.linalg.solve_discrete_are(a=sys.A.T, b=sys.C.T, q=Q, r=R)
    S = sys.C @ P @ sys.C.T + R
    return np.linalg.solve(S.T, sys.C @ P.T).T
