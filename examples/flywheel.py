#!/usr/bin/env python3

"""frccontrol example for a flywheel."""

import math
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import frccontrol as fct

if "--noninteractive" in sys.argv:
    mpl.use("svg")


class Flywheel:
    """An frccontrol system representing a flyhweel."""

    def __init__(self, dt):
        """Flywheel subsystem.

        Keyword arguments:
        dt -- time between model/controller updates
        """
        self.dt = dt

        # Number of motors
        num_motors = 1.0
        # Flywheel moment of inertia in kg-m^2
        J = 0.00032
        # Gear ratio
        G = 12.0 / 18.0
        self.plant = fct.models.flywheel(fct.models.MOTOR_775PRO, num_motors, J, G)

        # Sim variables
        self.sim = self.plant.to_discrete(self.dt)
        self.x = np.zeros((1, 1))
        self.u = np.zeros((1, 1))
        self.y = np.zeros((1, 1))

        # States: angular velocity (rad/s)
        # Inputs: voltage (V)
        # Outputs: angular velocity (rad/s)
        self.observer = fct.KalmanFilter(self.plant, [1.0], [0.01], self.dt)
        self.feedforward = fct.LinearPlantInversionFeedforward(
            self.plant.A, self.plant.B, self.dt
        )
        self.feedback = fct.LinearQuadraticRegulator(
            self.plant.A, self.plant.B, [9.42], [12.0], self.dt
        )

        self.u_min = np.array([[-12.0]])
        self.u_max = np.array([[12.0]])

    def update(self, r, next_r):
        """
        Advance the model by one timestep.

        Keyword arguments:
        r -- the current reference
        next_r -- the next reference
        """
        # Update sim model
        self.x = self.sim.A @ self.x + self.sim.B @ self.u
        self.y = self.sim.C @ self.x + self.sim.D @ self.u

        self.observer.predict(self.u, self.dt)
        self.observer.correct(self.u, self.y)
        self.u = np.clip(
            self.feedforward.calculate(next_r)
            + self.feedback.calculate(self.observer.x_hat, r),
            self.u_min,
            self.u_max,
        )


def main():
    """Entry point."""

    dt = 0.005
    flywheel = Flywheel(dt)

    # Set up graphing
    l0 = 0.1
    l1 = l0 + 5.0
    l2 = l1 + 0.1
    ts = np.arange(0, l2 + 5.0, dt)

    # Run simulation
    refs = []
    for t in ts:
        if t < l0:
            r = np.array([[0.0]])
        elif t < l1:
            r = np.array([[9000 / 60 * 2 * math.pi]])
        else:
            r = np.array([[0.0]])
        refs.append(r)
    r_rec, x_rec, u_rec, _ = fct.generate_time_responses(flywheel, refs)

    fct.plot_time_responses(
        ["Angular velocity (rad/s)"],
        ["Voltage (V)"],
        ts,
        r_rec,
        x_rec,
        u_rec,
    )
    if "--noninteractive" in sys.argv:
        plt.savefig("flywheel_response.svg")
    else:
        plt.show()


if __name__ == "__main__":
    main()
