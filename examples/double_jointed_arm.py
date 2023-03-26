#!/usr/bin/env python3

"""frccontrol example for a double-jointed arm."""

import sys

import matplotlib as mpl
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

import frccontrol as fct

if "--noninteractive" in sys.argv:
    mpl.use("svg")


class DoubleJointedArm:
    """
    An frccontrol system representing a double-jointed arm.

    States: [joint angle 1, joint angle 2,
             joint angular velocity 1, joint angular velocity 2,
             input error 1, input error 2]
    Inputs: [voltage 1, voltage 2]
    """

    def __init__(self, dt):
        """
        Double-jointed arm subsystem.

        Keyword arguments:
        dt -- simulation step time
        """
        self.dt = dt

        self.constants = DoubleJointedArmConstants()

        q_pos = 0.01745
        q_vel = 0.1745329
        q_error = 10
        r_pos = 0.05
        self.observer = fct.ExtendedKalmanFilter(
            6,
            2,
            self.f,
            self.h,
            [q_pos, q_pos, q_vel, q_vel, q_error, q_error],
            [r_pos, r_pos],
            self.dt,
        )

        self.x = np.zeros((6, 1))
        self.u = np.zeros((2, 1))
        self.y = np.zeros((2, 1))

        self.u_min = np.array([[-12.0]])
        self.u_max = np.array([[12.0]])

    # pragma pylint: disable=unused-argument
    def update(self, r, next_r):
        """
        Advance the model by one timestep.

        Keyword arguments:
        r -- the current reference
        next_r -- the next reference
        """
        self.x = fct.rkdp(self.f, self.x, self.u, self.dt)
        self.y = self.h(self.x, self.u)
        # self.y += np.array(
        #     [np.random.multivariate_normal(mean=[0, 0], cov=np.diag([1e-4, 1e-4]))]
        # ).T

        self.observer.predict(self.u, self.dt)
        self.observer.correct(self.u, self.y)

        rdot = (next_r - r) / self.dt

        u_ff = self.feedforward(r, rdot)

        A = fct.numerical_jacobian_x(
            6,
            6,
            self.f,
            self.observer.x_hat,
            self.feedforward(self.observer.x_hat, rdot),
        )
        B = fct.numerical_jacobian_u(
            6,
            2,
            self.f,
            self.observer.x_hat,
            self.feedforward(self.observer.x_hat, rdot),
        )
        u_fb = fct.LinearQuadraticRegulator(
            A[:4, :4],
            B[:4, :],
            [0.01745, 0.01745, 0.08726, 0.08726],
            [12.0, 12.0],
            self.dt,
        ).K @ (r[0:4] - self.observer.x_hat[0:4])

        # Voltage output from input error estimation. The estimate is in
        # Newton-meters, so has to be converted to volts.
        u_err = np.linalg.solve(self.constants.B, self.observer.x_hat[4:])

        self.u = np.clip(u_ff + u_fb - u_err, self.u_min, self.u_max)

    def get_dynamics_matrices(self, x):
        """Gets the dynamics matrices for the given state.

        See derivation at:
        https://www.chiefdelphi.com/t/whitepaper-two-jointed-arm-dynamics/423060

        Keyword arguments:
        x -- current system state
        """
        theta1, theta2, omega1, omega2 = x[:4].flat
        c2 = np.cos(theta2)

        l1 = self.constants.l1
        r1 = self.constants.r1
        r2 = self.constants.r2
        m1 = self.constants.m1
        m2 = self.constants.m2
        I1 = self.constants.I1
        I2 = self.constants.I2
        g = self.constants.g

        hM = l1 * r2 * c2
        M = (
            m1 * np.array([[r1 * r1, 0], [0, 0]])
            + m2
            * np.array(
                [[l1 * l1 + r2 * r2 + 2 * hM, r2 * r2 + hM], [r2 * r2 + hM, r2 * r2]]
            )
            + I1 * np.array([[1, 0], [0, 0]])
            + I2 * np.array([[1, 1], [1, 1]])
        )

        hC = -m2 * l1 * r2 * np.sin(theta2)
        C = np.array([[hC * omega2, hC * omega1 + hC * omega2], [-hC * omega1, 0]])

        G = (
            g * np.cos(theta1) * np.array([[m1 * r1 + m2 * self.constants.l1, 0]]).T
            + g * np.cos(theta1 + theta2) * np.array([[m2 * r2, m2 * r2]]).T
        )

        return M, C, G

    def f(self, x, u):
        """
        Dynamics model.

        Keyword arguments:
        x -- state vector
        u -- input vector
        """
        M, C, G = self.get_dynamics_matrices(x)

        omega = x[2:4]

        # Motor dynamics
        torque = self.constants.A @ x[2:4] + self.constants.B @ u

        # dx/dt = [ω α 0]ᵀ
        return np.block(
            [[x[2:4]], [np.linalg.solve(M, torque - C @ omega - G)], [np.zeros((2, 1))]]
        )

    def h(self, x, u):
        """
        Measurement model.

        Keyword arguments:
        x -- state vector
        u -- input vector
        """
        return x[:2, :]

    def feedforward(self, x, xdot):
        """
        Arm feedforward.
        """
        M, C, G = self.get_dynamics_matrices(x)
        omega = x[2:4]
        return np.linalg.solve(
            self.constants.B, M @ xdot[2:4] + C @ omega + G - self.constants.A @ omega
        )


class DoubleJointedArmConstants:
    """
    Double-jointed arm model constants.
    """

    def __init__(self):
        # Length of segments
        self.l1 = 46.25 * 0.0254
        self.l2 = 41.80 * 0.0254

        # Mass of segments
        self.m1 = 9.34 * 0.4536
        self.m2 = 9.77 * 0.4536

        # Distance from pivot to CG for each segment
        self.r1 = 21.64 * 0.0254
        self.r2 = 26.70 * 0.0254

        # Moment of inertia about CG for each segment
        self.I1 = 2957.05 * 0.0254 * 0.0254 * 0.4536
        self.I2 = 2824.70 * 0.0254 * 0.0254 * 0.4536

        # Gearing of each segment
        self.G1 = 70.0
        self.G2 = 45.0

        # Number of motors in each gearbox
        self.N1 = 1
        self.N2 = 2

        # Gravity
        self.g = 9.806

        motor = fct.models.MOTOR_NEO

        # torque = A * velocity + B * voltage
        self.B = (
            np.array([[self.N1 * self.G1, 0], [0, self.N2 * self.G2]])
            * motor.Kt
            / motor.R
        )
        self.A = (
            -np.array(
                [[self.G1 * self.G1 * self.N1, 0], [0, self.G2 * self.G2 * self.N2]]
            )
            * motor.Kt
            / motor.Kv
            / motor.R
        )

    def fwd_kinematics(self, x):
        """
        Forward kinematics for the given state.
        """
        theta1, theta2 = x[:2]

        joint2 = np.array([self.l1 * np.cos(theta1), self.l1 * np.sin(theta1)])
        end_eff = joint2 + np.array(
            [self.l2 * np.cos(theta1 + theta2), self.l2 * np.sin(theta1 + theta2)]
        )

        return joint2, end_eff

    def inv_kinematics(self, x, y, invert=False):
        """
        Inverse kinematics for a target position pos (x, y). The invert flag
        controls elbow direction.

        Keyword arguments:
        x -- x position
        y -- y position

        Returns:
        theta1 -- joint 1 angle
        theta2 -- joint 2 angle
        """
        theta2 = np.arccos(
            (x * x + y * y - (self.l1 * self.l1 + self.l2 * self.l2))
            / (2 * self.l1 * self.l2)
        )

        if invert:
            theta2 = -theta2

        theta1 = np.arctan2(y, x) - np.arctan2(
            self.l2 * np.sin(theta2), self.l1 + self.l2 * np.cos(theta2)
        )

        return theta1, theta2


def main():
    """Entry point."""

    dt = 0.02

    constants = DoubleJointedArmConstants()

    def to_state(x, y, invert):
        theta1, theta2 = constants.inv_kinematics(x, y, invert)
        return np.array([[theta1], [theta2], [0], [0]])

    state1 = to_state(1.5, -1, False)
    state2 = to_state(1.5, 1, True)
    state3 = to_state(-1.8, 1, False)

    traj1 = fct.trajectory.interpolate_states(0, 3, state1, state2)
    traj2 = fct.trajectory.interpolate_states(4, 8, state2, state3)
    traj = traj1.append(traj2)

    t_rec = np.arange(0, traj.end_time + dt + 1, dt)

    # Generate references for simulation
    refs = []
    for t in t_rec:
        r = np.block([[traj.sample(t)[:4]], [np.zeros((2, 1))]])
        refs.append(r)

    double_jointed_arm = DoubleJointedArm(dt)
    double_jointed_arm.x = refs[0]
    double_jointed_arm.observer.x_hat = refs[0]

    r_rec, x_rec, u_rec, _ = fct.generate_time_responses(double_jointed_arm, refs)
    fct.plot_time_responses(
        [
            "Angle 1 (rad)",
            "Angle 2 (rad)",
            "Angular velocity 1 (rad/s)",
            "Angular velocity 2 (rad/s)",
            "Input Error 1 (N-m)",
            "Input Error 2 (N-m)",
        ],
        ["Voltage 1 (V)", "Voltage 2 (V)"],
        t_rec,
        r_rec,
        x_rec,
        u_rec,
    )
    if "--noninteractive" in sys.argv:
        plt.savefig("double_jointed_arm_response.svg")
    else:
        animate_arm(double_jointed_arm, x_rec, r_rec)


def animate_arm(arm: DoubleJointedArm, x_rec, r_rec):
    """
    Animates arm.

    Keyword arguments:
    arm -- the arm being simulated
    x_rec -- state recording
    r_rec -- reference recording
    """

    def get_arm_joints(x):
        """
        Get the x-y positions of all three robot joints:

        - base joint
        - elbow
        - end effector
        """
        joint_pos, eff_pos = arm.constants.fwd_kinematics(x)
        return np.array([0, joint_pos[0, 0], eff_pos[0, 0]]), np.array(
            [0, joint_pos[1, 0], eff_pos[1, 0]]
        )

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.axis("square")

    total_len = arm.constants.l1 + arm.constants.l2
    ax.set_xlim(-total_len, total_len)
    ax.set_ylim(-total_len, total_len)

    xs, ys = get_arm_joints(x_rec[:, 0:1])
    arm_line = ax.plot(xs, ys, "-o", label="State")[0]

    xs, ys = get_arm_joints(r_rec[:, 0:1])
    ref_line = ax.plot(xs, ys, "--o", label="Reference")[0]

    ax.legend(loc="lower left")

    def animate(i):
        xs, ys = get_arm_joints(x_rec[:, i : i + 1])
        arm_line.set_data(xs, ys)

        xs, ys = get_arm_joints(r_rec[:, i : i + 1])
        ref_line.set_data(xs, ys)

        return arm_line, ref_line

    # pragma pylint: disable=unused-variable
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=x_rec.shape[1],
        interval=int(arm.dt * 1000),
        blit=True,
        repeat=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
