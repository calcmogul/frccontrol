"""
Extended Kalman filter class.
"""

import numpy as np
import scipy as sp

from .ctrlutil import make_cov_matrix, obsv
from .discretization import discretize_aq, discretize_r
from .numerical_integration import rkdp
from .numerical_jacobian import numerical_jacobian_x


class ExtendedKalmanFilter:
    """
    Extended Kalman filter class.
    """

    def __init__(
        self,
        states,
        inputs,
        f,
        h,
        state_std_devs: list[float],
        measurement_std_devs: list[float],
        dt: float,
    ):
        """
        Constructs an extended Kalman filter.

        Keyword arguments:
        states -- Number of states.
        inputs -- Number of inputs.
        f -- A vector-valued function of x and u that returns the derivative of
             the state vector.
        h -- A vector-valued function of x and u that returns the measurement
             vector.
        state_std_devs -- Standard deviations of model states.
        measurement_std_devs -- Standard deviations of measurements.
        dt -- Nominal discretization timestep.
        """
        self.states = states
        self.inputs = inputs
        self.outputs = len(measurement_std_devs)

        self.f = f
        self.h = h
        self.dt = dt
        self.x_hat = np.zeros((self.states, 1))

        self.contQ = make_cov_matrix(state_std_devs)
        self.contR = make_cov_matrix(measurement_std_devs)

        contA = numerical_jacobian_x(
            self.states, self.states, self.f, self.x_hat, np.zeros((inputs, 1))
        )
        C = numerical_jacobian_x(
            self.outputs, self.states, self.h, self.x_hat, np.zeros((inputs, 1))
        )

        discA, discQ = discretize_aq(contA, self.contQ, dt)
        discR = discretize_r(self.contR, dt)

        if (
            np.linalg.matrix_rank(obsv(discA, C)) == self.states
            and self.outputs <= self.states
        ):
            self.init_P = sp.linalg.solve_discrete_are(
                a=discA.T, b=C.T, q=discQ, r=discR
            )
        else:
            self.init_P = np.zeros((states, states))
        self.P = self.init_P

    def reset(self):
        """
        Resets the observer.
        """
        self.x_hat = np.zeros(self.x_hat.shape)
        self.P = self.init_P

    def predict(self, u, dt):
        """
        Project the model into the future with a new control input u.

        Keyword arguments:
        u -- New control input from controller.
        dt -- Timestep for prediction.
        """
        # Find continuous A
        contA = numerical_jacobian_x(
            self.states,
            self.states,
            self.f,
            self.x_hat,
            u,
        )

        # Find discrete A and Q
        discA, discQ = discretize_aq(contA, self.contQ, dt)

        self.x_hat = rkdp(self.f, self.x_hat, u, dt)

        # Pₖ₊₁⁻ = APₖ⁻Aᵀ + Q
        self.P = discA @ self.P @ discA.T + discQ

        self.dt = dt

    def correct(self, u, y):
        """
        Correct the state estimate x-hat using the measurements in y.

        Keyword arguments:
        u -- Same control input used in the last predict step.
        y -- Measurement vector.
        """
        C = numerical_jacobian_x(self.outputs, self.states, self.h, self.x_hat, u)
        R = discretize_r(self.contR, self.dt)

        S = C @ self.P @ C.T + R

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
        K = np.linalg.solve(S.T, C @ self.P.T).T

        # x̂ₖ₊₁⁺ = x̂ₖ₊₁⁻ + Kₖ₊₁(y − h(x̂ₖ₊₁⁻, uₖ₊₁))
        self.x_hat += K @ (y - self.h(self.x_hat, u))

        # Pₖ₊₁⁺ = (I−Kₖ₊₁C)Pₖ₊₁⁻(I−Kₖ₊₁C)ᵀ + Kₖ₊₁RKₖ₊₁ᵀ
        # Use Joseph form for numerical stability
        self.P = (np.eye(self.states) - K @ C) @ self.P @ (
            np.eye(self.states) - K @ C
        ).T + K @ R @ K.T
