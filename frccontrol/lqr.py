import control as ct
import numpy as np
import scipy as sp


def lqr(*args, **kwargs):
    """Solves for the optimal linear-quadratic regulator (LQR).

    For a continuous system:

    .. math:: xdot = A * x + B * u
    .. math:: J = \\int_0^\\infty (x^T Q x + u^T R u + 2 x^T N u) dt

    For a discrete system:

    .. math:: x(n + 1) = A x(n) + B u(n)
    .. math:: J = \\sum_0^\\infty (x^T Q x + u^T R u + 2 x^T N u) \\Delta T

    Keyword arguments:
    sys -- StateSpace object representing a linear system.
    Q -- numpy.array(states x states), state cost matrix.
    R -- numpy.array(inputs x inputs), control effort cost matrix.
    N -- numpy.array(states x inputs), cross weight matrix.

    Returns:
    K -- numpy.array(states x inputs), controller gain matrix.
    """
    sys = args[0]
    Q = args[1]
    R = args[2]
    if len(args) == 4:
        N = args[3]
    else:
        N = np.zeros((sys.A.shape[0], sys.B.shape[1]))

    if np.linalg.matrix_rank(ct.ctrb(sys.A, sys.B)) < sys.A.shape[0]:
        print(f"Warning: The system is uncontrollable\n\nA = {sys.A}\nB = {sys.B}\n")

    if sys.dt == None:
        P = sp.linalg.solve_continuous_are(a=sys.A, b=sys.B, q=Q, r=R, s=N)
        return np.linalg.solve(R, sys.B.T @ P + N.T)
    else:
        P = sp.linalg.solve_discrete_are(a=sys.A, b=sys.B, q=Q, r=R, s=N)
        return np.linalg.solve(R + sys.B.T @ P @ sys.B, sys.B.T @ P @ sys.A + N.T)
