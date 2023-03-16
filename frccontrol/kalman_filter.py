"""
Linear Kalman filter class.
"""

import numpy as np
import scipy as sp

from .ctrlutil import make_cov_matrix, obsv
from .discretization import discretize_ab, discretize_aq, discretize_r


class KalmanFilter:
    """
    Linear Kalman filter class.
    """

    def __init__(
        self,
        plant: sp.signal.StateSpace,
        state_std_devs: list[float],
        measurement_std_devs: list[float],
        dt: float,
    ):
        """
        Constructs a Kalman filter.

        Keyword arguments:
        plant -- The plant used for the prediction step.
        state_std_devs -- Standard deviations of model states.
        measurement_std_devs -- Standard deviations of measurements.
        dt -- Nominal discretization timestep.
        """
        self.plant = plant

        A, Q = discretize_aq(plant.A, make_cov_matrix(state_std_devs), dt)
        C = plant.C
        R = discretize_r(make_cov_matrix(measurement_std_devs), dt)

        if np.linalg.matrix_rank(obsv(A, C)) < A.shape[0]:
            print(f"Warning: The system is unobservable\n\nA = {A}\nC = {C}\n")

        P = sp.linalg.solve_discrete_are(a=A.T, b=C.T, q=Q, r=R)

        # S = CPCᵀ + R
        S = C @ P @ C.T + R

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
        self.K = np.linalg.solve(S.T, C @ P.T).T

        self.x_hat = np.zeros((self.plant.A.shape[0], 1))

    def reset(self):
        """
        Resets the observer.
        """
        self.x_hat = np.zeros(self.x_hat.shape)

    def predict(self, u, dt):
        """
        Project the model into the future with a new control input u.

        Keyword arguments:
        u -- New control input from controller.
        dt -- Timestep for prediction.
        """
        A, B = discretize_ab(self.plant.A, self.plant.B, dt)
        self.x_hat = A @ self.x_hat + B @ u

    def correct(self, u, y):
        """
        Correct the state estimate x-hat using the measurements in y.

        Keyword arguments:
        u -- Same control input used in the last predict step.
        y -- Measurement vector.
        """
        # x̂ₖ₊₁⁺ = x̂ₖ₊₁⁻ + K(y − (Cx̂ₖ₊₁⁻ + Duₖ₊₁))
        self.x_hat += self.K @ (y - (self.plant.C @ self.x_hat + self.plant.D @ u))
