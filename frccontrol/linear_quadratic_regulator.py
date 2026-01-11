"""
Linear quadratic regulator class.
"""

import numpy as np
import scipy as sp

from .ctrlutil import cost_matrix, is_stabilizable
from .discretization import discretize_ab


class LinearQuadraticRegulator:
    """Linear-quadratic regulator class."""

    def __init__(self, A, B, Qelems, Relems, dt):
        """
        Constructs a LinearQuadraticRegulator.

        Parameter ``A``:
            Continuous system matrix.

        Parameter ``B``:
            Continuous input matrix.

        Parameter ``Qelems``:
            The maximum desired error tolerance for each state.

        Parameter ``Relems``:
            The maximum desired control effort for each input.

        Parameter ``dt``:
            Discretization timestep.
        """
        self.states = A.shape[0]
        self.inputs = B.shape[1]

        discA, discB = discretize_ab(A, B, dt)

        if not is_stabilizable(discA, discB):
            raise RuntimeError(
                f"The system is unstabilizable!\n\nA = {discA}\nB = {discB}\n"
            )

        Q = cost_matrix(Qelems)
        R = cost_matrix(Relems)

        S = sp.linalg.solve_discrete_are(a=discA, b=discB, q=Q, r=R)

        # K = (BᵀSB + R)⁻¹BᵀSA
        self.K = np.linalg.solve(discB.T @ S @ discB + R, discB.T @ S @ discA)

        self.r = np.zeros((self.states, 1))
        self.u = np.zeros((self.inputs, 1))

    def reset(self):
        """Resets the controller."""
        self.r = np.zeros((self.states, 1))
        self.u = np.zeros((self.inputs, 1))

    def calculate(self, x, r=None):
        """
        Returns the next output of the controller.

        Parameter ``x``:
            The current state vector x.

        Parameter ``r``:
            The current reference vector r (default: previous reference).
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

        Parameter ``A``:
            Continuous system matrix.

        Parameter ``B``:
            Continuous input matrix.

        Parameter ``dt``:
            Discretization timestep.

        Parameter ``input_delay``:
            Input time delay.
        """
        discA, discB = discretize_ab(A, B, dt)
        self.K = self.K @ sp.linalg.fractional_matrix_power(
            discA - discB @ self.K, input_delay / dt
        )
