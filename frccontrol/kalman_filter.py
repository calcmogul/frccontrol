"""
Linear Kalman filter class.
"""

import numpy as np
import scipy as sp

from .ctrlutil import covariance_matrix, is_detectable
from .discretization import discretize_ab, discretize_aq, discretize_r


class KalmanFilter:
    """Linear Kalman filter class."""

    def __init__(
        self,
        plant: sp.signal.StateSpace,
        state_std_devs: list[float],
        measurement_std_devs: list[float],
        dt: float,
    ):
        """
        Constructs a Kalman filter.

        Parameter ``plant``:
            The plant used for the prediction step.

        Parameter ``state_std_devs``:
            Standard deviations of model states.

        Parameter ``measurement_std_devs``:
            Standard deviations of measurements.

        Parameter ``dt``:
            Nominal discretization timestep.
        """
        self.plant = plant

        self.contQ = covariance_matrix(state_std_devs)
        self.contR = covariance_matrix(measurement_std_devs)
        self.dt = dt

        # Find discrete A and Q
        discA, discQ = discretize_aq(plant.A, self.contQ, dt)

        discR = discretize_r(self.contR, dt)

        C = plant.C

        if not is_detectable(discA, C):
            raise RuntimeError(f"The system is undetectable!\n\nA = {discA}\nC = {C}\n")

        self.P = sp.linalg.solve_discrete_are(a=discA.T, b=C.T, q=discQ, r=discR)

        self.x_hat = np.zeros((self.plant.A.shape[0], 1))

    def reset(self):
        """
        Resets the observer.
        """
        self.x_hat = np.zeros(self.x_hat.shape)

    def predict(self, u, dt):
        """
        Project the model into the future with a new control input u.

        Parameter ``u``:
            New control input from controller.

        Parameter ``dt``:
            Timestep for prediction.
        """
        discA, discB = discretize_ab(self.plant.A, self.plant.B, dt)
        _, discQ = discretize_aq(self.plant.A, self.contQ, dt)

        # x̂ₖ₊₁⁻ = Ax̂ₖ⁺ + Buₖ
        self.x_hat = discA @ self.x_hat + discB @ u

        # Pₖ₊₁⁻ = APₖ⁻Aᵀ + Q
        self.P = discA @ self.P @ discA.T + discQ

        self.dt = dt

    def correct(self, u, y):
        """
        Correct the state estimate x-hat using the measurements in y.

        Parameter ``u``:
            Same control input used in the last predict step.

        Parameter ``y``:
            Measurement vector.
        """
        C = self.plant.C
        D = self.plant.D

        discR = discretize_r(self.contR, self.dt)

        S = C @ self.P @ C.T + discR

        # We want to put K = PCᵀS⁻¹ into Ax = b form so we can solve it more
        # efficiently.
        #
        # K = PCᵀS⁻¹
        # KS = PCᵀ
        # (KS)ᵀ = (PCᵀ)ᵀ
        # SᵀKᵀ = CPᵀ
        #
        # The solution of Ax = b can be found via x = A.solve(b).
        #
        # Kᵀ = Sᵀ.solve(CPᵀ)
        # K = (Sᵀ.solve(CPᵀ))ᵀ
        #
        # Drop the transposes on symmetric matrices S and P.
        #
        # K = (S.solve(CP))ᵀ
        K = np.linalg.solve(S, C @ self.P).T

        # x̂ₖ₊₁⁺ = x̂ₖ₊₁⁻ + K(y − (Cx̂ₖ₊₁⁻ + Duₖ₊₁))
        self.x_hat += K @ (y - (C @ self.x_hat + D @ u))

        # Pₖ₊₁⁺ = (I−Kₖ₊₁C)Pₖ₊₁⁻(I−Kₖ₊₁C)ᵀ + Kₖ₊₁RKₖ₊₁ᵀ
        # Use Joseph form for numerical stability
        I = np.eye(self.plant.A.shape[0])
        self.P = (I - K @ C) @ self.P @ (I - K @ C).T + K @ discR @ K.T
