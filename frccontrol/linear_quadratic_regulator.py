"""
Linear quadratic regulator class.
"""

import numpy as np
import scipy as sp

from .ctrlutil import ctrb, make_cost_matrix
from .discretization import discretize_ab


class LinearQuadraticRegulator:
    """
    Linear quadratic regulator class.
    """

    def __init__(self, A, B, Qelems, Relems, dt):
        """
        Constructs a LinearQuadraticRegulator.

        Keyword arguments:
        A -- Continuous system matrix.
        B -- Continuous input matrix.
        Qelems -- The maximum desired error tolerance for each state.
        Relems -- The maximum desired control effort for each input.
        dt -- Discretization timestep.
        """
        self.states = A.shape[0]
        self.inputs = B.shape[1]

        discA, discB = discretize_ab(A, B, dt)

        if np.linalg.matrix_rank(ctrb(discA, discB)) < discA.shape[0]:
            print(
                f"Warning: The system is uncontrollable\n\nA = {discA}\nB = {discB}\n"
            )

        Q = make_cost_matrix(Qelems)
        R = make_cost_matrix(Relems)

        S = sp.linalg.solve_discrete_are(a=discA, b=discB, q=Q, r=R)

        # K = (BᵀSB + R)⁻¹BᵀSA
        self.K = np.linalg.solve(discB.T @ S @ discB + R, discB.T @ S @ discA)

        self.r = np.zeros((self.states, 1))
        self.u = np.zeros((self.inputs, 1))

    def reset(self):
        """
        Resets the controller.
        """
        self.r = np.zeros((self.states, 1))
        self.u = np.zeros((self.inputs, 1))

    def calculate(self, x, r=None):
        """
        Returns the next output of the controller.

        Keyword arguments:
        x -- The current state vector x.
        r -- The current reference vector r (default: previous reference)
        """
        if r is not None:
            self.r = r
        self.u = self.K @ (self.r - x)
        return self.u

    def latency_compensate(self, A, B, dt, input_delay):
        """
        Adjusts LQR controller gain to compensate for a pure time delay in the
        input.

        Linear-Quadratic regulator controller gains tend to be aggressive. If
        sensor measurements are time-delayed too long, the LQR may be unstable.
        However, if we know the amount of delay, we can compute the control
        based on where the system will be after the time delay.

        See https://file.tavsys.net/control/controls-engineering-in-frc.pdf
        appendix C.4 for a derivation.

        Keyword arguments:
        A -- Continuous system matrix.
        B -- Continuous input matrix.
        dt -- Discretization timestep.
        input_delay -- Input time delay.
        """
        discA, discB = discretize_ab(A, B, dt)
        self.K = self.K @ sp.linalg.fractional_matrix_power(
            discA - discB @ self.K, input_delay / dt
        )
