"""
Extended Kalman filter class.
"""

import numpy as np
import scipy as sp

from .ctrlutil import covariance_matrix, is_detectable
from .discretization import discretize_aq, discretize_r
from .numerical_integration import rkdp
from .numerical_jacobian import numerical_jacobian_x


class ExtendedKalmanFilter:
    """Extended Kalman filter class."""

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

        Parameter ``states``:
            Number of states.
        Parameter ``inputs``:
            Number of inputs.
        Parameter ``f``:
            A vector-valued function of x and u that returns the derivative of
            the state vector.
        Parameter ``h``:
            A vector-valued function of x and u that returns the measurement
            vector.
        Parameter ``state_std_devs``:
            Standard deviations of model states.
        Parameter ``measurement_std_devs``:
            Standard deviations of measurements.
        Parameter ``dt``:
            Nominal discretization timestep.
        """
        self.states = states
        self.inputs = inputs
        self.outputs = len(measurement_std_devs)

        self.f = f
        self.h = h
        self.dt = dt
        self.x_hat = np.zeros((self.states, 1))

        self.contQ = covariance_matrix(state_std_devs)
        self.contR = covariance_matrix(measurement_std_devs)

        contA = numerical_jacobian_x(
            self.states, self.states, self.f, self.x_hat, np.zeros((inputs, 1))
        )
        C = numerical_jacobian_x(
            self.outputs, self.states, self.h, self.x_hat, np.zeros((inputs, 1))
        )

        discA, discQ = discretize_aq(contA, self.contQ, dt)
        discR = discretize_r(self.contR, dt)

        if is_detectable(discA, C) and self.outputs <= self.states:
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

        Parameter ``u``:
            New control input from controller.
        Parameter ``dt``:
            Timestep for prediction.
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

        Parameter ``u``:
            Same control input used in the last predict step.
        Parameter ``y``:
            Measurement vector.
        """
        C = numerical_jacobian_x(self.outputs, self.states, self.h, self.x_hat, u)
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

        # x̂ₖ₊₁⁺ = x̂ₖ₊₁⁻ + Kₖ₊₁(y − h(x̂ₖ₊₁⁻, uₖ₊₁))
        self.x_hat += K @ (y - self.h(self.x_hat, u))

        # Pₖ₊₁⁺ = (I−Kₖ₊₁C)Pₖ₊₁⁻(I−Kₖ₊₁C)ᵀ + Kₖ₊₁RKₖ₊₁ᵀ
        # Use Joseph form for numerical stability
        I = np.eye(self.states)
        self.P = (I - K @ C) @ self.P @ (I - K @ C).T + K @ discR @ K.T
