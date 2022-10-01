#!/usr/bin/env python3

"""frccontrol example for an elevator."""

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import frccontrol as fct

if "--noninteractive" in sys.argv:
    mpl.use("svg")


class Elevator(fct.System):
    """An frccontrol system representing an elevator."""

    def __init__(self, dt):
        """Elevator subsystem.

        Keyword arguments:
        dt -- time between model/controller updates
        """
        state_labels = [("Position", "m"), ("Velocity", "m/s")]
        u_labels = [("Voltage", "V")]
        self.set_plot_labels(state_labels, u_labels)

        fct.System.__init__(
            self,
            np.array([[-12.0]]),
            np.array([[12.0]]),
            dt,
            np.zeros((2, 1)),
            np.zeros((1, 1)),
        )

    # pragma pylint: disable=signature-differs
    def create_model(self, states, inputs):
        # Number of motors
        num_motors = 2.0
        # Elevator carriage mass in kg
        m = 6.803886
        # Radius of pulley in meters
        r = 0.02762679089
        # Gear ratio
        G = 42.0 / 12.0 * 40.0 / 14.0

        return fct.models.elevator(fct.models.MOTOR_CIM, num_motors, m, r, G)

    def design_controller_observer(self):
        q = [0.02, 0.4]
        r = [12.0]
        self.design_lqr(q, r)
        self.design_two_state_feedforward()

        q_pos = 0.05
        q_vel = 1.0
        r_pos = 0.0001
        self.design_kalman_filter([q_pos, q_vel], [r_pos])


def main():
    """Entry point."""

    dt = 0.005
    elevator = Elevator(dt)

    # Set up graphing
    l0 = 0.1
    l1 = l0 + 5.0
    l2 = l1 + 0.1
    ts = np.arange(0, l2 + 5.0, dt)

    refs = []

    # Generate references for simulation
    for t in ts:
        if t < l0:
            r = np.array([[0.0], [0.0]])
        elif t < l1:
            r = np.array([[1.524], [0.0]])
        else:
            r = np.array([[0.0], [0.0]])
        refs.append(r)

    x_rec, ref_rec, u_rec, _ = elevator.generate_time_responses(refs)
    elevator.plot_time_responses(ts, x_rec, ref_rec, u_rec)
    if "--noninteractive" in sys.argv:
        plt.savefig("elevator_response.svg")
    else:
        plt.show()


if __name__ == "__main__":
    main()
