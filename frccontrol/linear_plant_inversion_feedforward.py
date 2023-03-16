"""
Linear plant inversion feedforward.
"""

import numpy as np

from .discretization import discretize_ab


class LinearPlantInversionFeedforward:
    """
    A linear plant inversion feedforward.
    """

    def __init__(self, A, B, dt):
        """
        Constructs a linear plant inversion feedforward.

        Keyword arguments:
        A -- Continuous system matrix of the plant being controlled.
        B -- Continuous input matrix of the plant being controlled.
        dt -- Discretization timestep.
        """
        self.A, self.B = discretize_ab(A, B, dt)
        self.r = np.zeros((A.shape[0], 1))
        self.u_ff = np.zeros((B.shape[1], 1))

    def reset(self, initial_reference):
        """
        Resets the feedforward with a specified initial reference vector.

        Keyword arguments:
        initial_reference -- The initial reference vector.
        """
        self.r = initial_reference
        self.u_ff = np.zeros(self.u_ff.shape)

    def calculate(self, next_r):
        """
        Calculate the feedforward with only the desired future reference. This
        uses the internally stored "current" reference (calling reset() will
        override this).

        Keyword arguments:
        next_r -- The reference state of the future timestep (k + 1).

        Returns:
        The calculated feedforward.
        """
        self.u_ff = np.linalg.pinv(self.B) @ (next_r - (self.A @ self.r))
        self.r = next_r
        return self.u_ff
