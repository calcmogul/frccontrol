#!/usr/bin/env python3

# Avoid needing display if plots aren't being shown
import sys

if "--noninteractive" in sys.argv:
    import matplotlib as mpl

    mpl.use("svg")

import frccontrol as frccnt
import matplotlib.pyplot as plt
import numpy as np


class Drivetrain(frccnt.System):
    def __init__(self, dt):
        """Drivetrain subsystem.

        Keyword arguments:
        dt -- time between model/controller updates
        """
        state_labels = [
            ("Left position", "m"),
            ("Left velocity", "m/s"),
            ("Right position", "m"),
            ("Right velocity", "m/s"),
        ]
        u_labels = [("Left voltage", "V"), ("Right voltage", "V")]
        self.set_plot_labels(state_labels, u_labels)

        self.in_low_gear = False

        # Number of motors per side
        self.num_motors = 2.0

        # High and low gear ratios of drivetrain
        Glow = 60.0 / 11.0
        Ghigh = 60.0 / 11.0

        # Drivetrain mass in kg
        self.m = 52
        # Radius of wheels in meters
        self.r = 0.08255 / 2.0
        # Radius of robot in meters
        self.rb = 0.59055 / 2.0
        # Moment of inertia of the drivetrain in kg-m^2
        self.J = 6.0

        # Gear ratios of left and right sides of drivetrain respectively
        if self.in_low_gear:
            self.Gl = Glow
            self.Gr = Glow
        else:
            self.Gl = Ghigh
            self.Gr = Ghigh

        self.model = frccnt.models.drivetrain(
            frccnt.models.MOTOR_CIM,
            self.num_motors,
            self.m,
            self.r,
            self.rb,
            self.J,
            self.Gl,
            self.Gr,
        )
        u_min = np.matrix([[-12.0], [-12.0]])
        u_max = np.matrix([[12.0], [12.0]])
        frccnt.System.__init__(self, self.model, u_min, u_max, dt)

        if self.in_low_gear:
            q_pos = 0.12
            q_vel = 1.0
        else:
            q_pos = 0.14
            q_vel = 0.95

        q = [q_pos, q_vel, q_pos, q_vel]
        r = [12.0, 12.0]
        self.design_dlqr_controller(q, r)

        qff_pos = 0.005
        qff_vel = 1.0
        self.design_two_state_feedforward(
            [qff_pos, qff_vel, qff_pos, qff_vel], [12.0, 12.0]
        )

        q_pos = 0.05
        q_vel = 1.0
        q_voltage = 10.0
        q_encoder_uncertainty = 2.0
        r_pos = 0.0001
        r_gyro = 0.000001
        self.design_kalman_filter([q_pos, q_vel, q_pos, q_vel], [r_pos, r_pos])


def main():
    dt = 0.00505
    drivetrain = Drivetrain(dt)
    drivetrain.export_cpp_coeffs("Drivetrain", "Subsystems/")

    if "--save-plots" in sys.argv or "--noninteractive" not in sys.argv:
        try:
            import slycot

            plt.figure(1)
            drivetrain.plot_pzmaps()
        except ImportError:  # Slycot unavailable. Can't show pzmaps.
            pass
    if "--save-plots" in sys.argv:
        plt.savefig("drivetrain_pzmaps.svg")

    t, xprof, vprof, aprof = frccnt.generate_s_curve_profile(
        max_v=4.0, max_a=3.5, time_to_max_a=1.0, dt=dt, goal=50.0
    )

    # Generate references for simulation
    refs = []
    for i in range(len(t)):
        r = np.matrix([[xprof[i]], [vprof[i]], [xprof[i]], [vprof[i]]])
        refs.append(r)

    if "--save-plots" in sys.argv or "--noninteractive" not in sys.argv:
        plt.figure(2)
        state_rec, ref_rec, u_rec = drivetrain.generate_time_responses(t, refs)
        drivetrain.plot_time_responses(t, state_rec, ref_rec, u_rec)
    if "--save-plots" in sys.argv:
        plt.savefig("drivetrain_response.svg")
    if "--noninteractive" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
