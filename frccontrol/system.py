"""A class that simplifies creating and updating state-space models as well as
designing controllers for them.
"""

from abc import abstractmethod, ABCMeta
import control as ct
import frccontrol as fct
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


class System:
    __metaclass__ = ABCMeta

    def __init__(self, u_min, u_max, dt, states, inputs, nonlinear_func=None):
        """Sets up the matrices for a state-space model.

        Keyword arguments:
        u_min -- vector of minimum control inputs for system
        u_max -- vector of maximum control inputs for system
        dt -- time between model/controller updates
        states -- initial state vector around which to linearize model
        inputs -- initial input vector around which to linearize model
        nonlinear_func -- function that takes x and u and returns the state
                          derivative for a nonlinear system (optional)
        """
        self.f = nonlinear_func
        self.sysc = self.create_model(np.asarray(states), np.asarray(inputs))
        self.dt = dt
        self.sysd = self.sysc.sample(self.dt)  # Discretize model

        # Model matrices
        self.x = np.asarray(states)
        self.u = np.asarray(inputs)
        self.y = np.zeros((self.sysc.C.shape[0], 1))

        # Controller matrices
        self.r = np.zeros((self.sysc.A.shape[0], 1))

        # Observer matrices
        self.x_hat = np.asarray(states)

        self.u_min = np.asarray(u_min)
        self.u_max = np.asarray(u_max)

        # Controller matrices
        self.K = np.zeros((self.sysc.B.shape[1], self.sysc.B.shape[0]))
        self.Kff = np.zeros((self.sysc.B.shape[1], self.sysc.B.shape[0]))

        # Observer matrices
        self.P = np.zeros(self.sysc.A.shape)
        self.kalman_gain = np.zeros((self.sysc.A.shape[0], self.sysc.C.shape[0]))

        self.design_controller_observer()

    __default = object()

    def update(self, next_r=__default):
        """Advance the model by one timestep.

        Keyword arguments:
        next_r -- next controller reference (default: current reference)
        """
        self.update_plant()
        self.correct_observer()
        self.update_controller(next_r)
        self.predict_observer()

    def update_plant(self):
        """Advance the model by one timestep."""
        if self.f:
            from . import runge_kutta

            self.x = runge_kutta(self.f, self.x, self.u, self.dt)
        else:
            self.x = self.sysd.A @ self.x + self.sysd.B @ self.u
        self.y = self.sysd.C @ self.x + self.sysd.D @ self.u

    def predict_observer(self):
        """Runs the predict step of the observer update.

        In one update step, this should be run after correct_observer().
        """
        if self.f:
            from . import runge_kutta

            self.x_hat = runge_kutta(self.f, self.x, self.u, self.dt)
            self.P = self.sysd.A @ self.P @ self.sysd.A.T + self.Q
        else:
            self.x_hat = self.sysd.A @ self.x_hat + self.sysd.B @ self.u

    def correct_observer(self):
        """Runs the correct step of the observer update.

        In one update step, this should be run before predict_observer().
        """
        if self.f:
            self.kalman_gain = (
                self.P
                @ self.sysd.C.T
                @ np.linalg.inv(self.sysd.C @ self.P @ self.sysd.C.T + self.R)
            )
            self.P = (
                np.eye(self.sysd.A.shape[0]) - self.kalman_gain @ self.sysd.C
            ) @ self.P
        self.x_hat += self.kalman_gain @ (
            self.y - self.sysd.C @ self.x_hat - self.sysd.D @ self.u
        )

    def update_controller(self, next_r=__default):
        """Advance the controller by one timestep.

        Keyword arguments:
        next_r -- next controller reference (default: current reference)
        """
        u = self.K @ (self.r - self.x_hat)
        if next_r is not self.__default:
            if self.f:
                rdot = (next_r - self.r) / self.dt
                uff = self.Kff @ (rdot - self.f(self.r, np.zeros(self.u.shape)))
            else:
                uff = self.Kff @ (next_r - self.sysd.A @ self.r)
            self.r = next_r
        else:
            if self.f:
                uff = -self.Kff @ self.f(self.r, np.zeros(self.u.shape))
            else:
                uff = self.Kff @ (self.r - self.sysd.A @ self.r)
        self.u = np.clip(u + uff, self.u_min, self.u_max)

    @abstractmethod
    def create_model(self, states=__default, inputs=__default):
        """Relinearize model around given state.

        Keyword arguments:
        states -- state vector around which to linearize model (if applicable)
        inputs -- input vector around which to linearize model (if applicable)

        Returns:
        StateSpace instance containing continuous state-space model
        """
        return

    @abstractmethod
    def design_controller_observer(self):
        pass

    def design_lqr(self, Q_elems, R_elems):
        """Design a discrete time linear-quadratic regulator for the system.

        Keyword arguments:
        Q_elems -- a vector of the maximum allowed excursions of the states from
                   the reference.
        R_elems -- a vector of the maximum allowed excursions of the control
                   inputs from no actuation.
        """
        from . import lqr

        Q = self.__make_cost_matrix(Q_elems)
        R = self.__make_cost_matrix(R_elems)
        self.K = lqr(self.sysd, Q, R)

    def place_controller_poles(self, poles):
        """Design a controller that places the closed-loop system poles at the
        given locations.

        Most users should just use design_dlqr_controller(). Only use this if
        you know what you're doing.

        Keyword arguments:
        poles -- a list of compex numbers which are the desired pole locations.
                 Complex conjugate poles must be in pairs.
        """
        self.K = ct.place(self.sysd.A, self.sysd.B, poles)

    def design_kalman_filter(self, Q_elems, R_elems):
        """Design a discrete time Kalman filter for the system.

        Keyword arguments:
        Q_elems -- a vector of the standard deviations of each state from how
                   the model behaves.
        R_elems -- a vector of the standard deviations of each output
                   measurement.
        """
        self.Q = self.__make_cov_matrix(Q_elems)
        self.R = self.__make_cov_matrix(R_elems)
        if not self.f:
            from . import kalmd

            self.kalman_gain = kalmd(self.sysd, Q=self.Q, R=self.R)
        else:
            m = self.sysd.A.shape[0]
            M = np.concatenate(
                (
                    np.concatenate((-self.sysd.A.T, self.Q), axis=1),
                    np.concatenate((np.zeros(self.sysd.A.shape), self.sysd.A), axis=1),
                ),
                axis=0,
            )
            M = sp.linalg.expm(M * self.sysd.dt)
            self.Q = M[m:, m:] @ M[:m, m:]
            self.R = 1 / self.sysd.dt * self.R

    def place_observer_poles(self, poles):
        """Design a controller that places the closed-loop system poles at the
        given locations.

        Most users should just use design_kalman_filter(). Only use this if you
        know what you're doing.

        Keyword arguments:
        poles -- a list of compex numbers which are the desired pole locations.
                 Complex conjugate poles must be in pairs.
        """
        L = ct.place(self.sysd.A.T, self.sysd.C.T, poles).T
        self.kalman_gain = np.linalg.inv(self.sysd.A) @ L

    def design_two_state_feedforward(self, Q_elems=None, R_elems=None):
        """Computes the feedforward constant for a two-state controller.

        This will take the form u = K_ff * (r_{n+1} - A r_n), where K_ff is the
        feed-forwards constant. It is important that Kff is *only* computed off
        the goal and not the feedback terms.

        If either Q_elems or R_elems is not specified, then both are ignored.

        Keyword arguments:
        Q_elems -- a vector of the maximum allowed excursions in the state
                   tracking.
        R_elems -- a vector of the maximum allowed excursions of the control
                   inputs from no actuation.
        """
        if Q_elems is not None and R_elems is not None:
            # We want to find the optimal U such that we minimize the tracking
            # cost. This means that we want to minimize
            #   (B u - (r_{n+1} - A r_n))^T Q (B u - (r_{n+1} - A r_n)) + u^T R u
            Q = self.__make_cost_matrix(Q_elems)
            R = self.__make_cost_matrix(R_elems)
            if self.f:
                self.Kff = (
                    np.linalg.inv(self.sysc.B.T @ Q @ self.sysc.B + R.T)
                    @ self.sysc.B.T
                    @ Q
                )
            else:
                self.Kff = (
                    np.linalg.inv(self.sysd.B.T @ Q @ self.sysd.B + R.T)
                    @ self.sysd.B.T
                    @ Q
                )
        else:
            # Without Q and R weighting matrices, K_ff = B^+ where B^+ is the
            # Moore-Penrose pseudoinverse of B.
            if self.f:
                self.Kff = np.linalg.pinv(self.sysc.B)
            else:
                self.Kff = np.linalg.pinv(self.sysd.B)

    def plot_pzmaps(self, discrete=True):
        """Plots pole-zero maps of open-loop system, closed-loop system, and
        observer poles.

        Keyword arguments:
        discrete -- whether to make pole-zero map of continuous or discrete
                    version of system
        """
        fct.plot_open_loop_poles(self, discrete)
        fct.plot_closed_loop_poles(self, discrete)
        fct.plot_observer_poles(self, discrete)
        plt.tight_layout()

    def extract_row(self, buf, idx):
        """Extract row from 2D array.

        Keyword arguments:
        buf -- matrix containing plot data
        idx -- index of desired plot in buf

        Returns:
        Desired list of data from buf
        """
        return np.squeeze(buf[idx : idx + 1, :])

    def generate_time_responses(self, t, refs):
        """Generate time-domain responses of the system and the control inputs.

        Returns:
        x_rec -- recording of state estimates
        ref_rec -- recording of references
        u_rec -- recording of inputs
        y_rec -- recording of outputs

        Keyword arguments:
        t -- list of timesteps corresponding to references
        refs -- list of reference vectors, one for each time
        """
        x_rec = np.zeros((self.sysd.states, 0))
        ref_rec = np.zeros((self.sysd.states, 0))
        u_rec = np.zeros((self.sysd.inputs, 0))
        y_rec = np.zeros((self.sysd.outputs, 0))

        # Run simulation
        self.r = refs[0]
        for i in range(len(refs)):
            next_r = refs[i]
            self.update(next_r)

            # Log states for plotting
            x_rec = np.concatenate((x_rec, self.x_hat), axis=1)
            ref_rec = np.concatenate((ref_rec, self.r), axis=1)
            u_rec = np.concatenate((u_rec, self.u), axis=1)
            y_rec = np.concatenate((y_rec, self.y), axis=1)

        return x_rec, ref_rec, u_rec, y_rec

    def plot_time_responses(self, t, x_rec, ref_rec, u_rec, title=__default):
        """Plots time-domain responses of the system and the control inputs.

        Keyword arguments:
        time -- list of timesteps corresponding to references
        x_rec -- recording of state estimates from generate_time_responses()
        ref_rec -- recording of references from generate_time_responses()
        u_rec -- recording of inputs from generate_time_responses()
        title -- title for time-domain plots (default: "Time-domain responses")
        """
        plt.figure()
        subplot_max = self.sysd.states + self.sysd.inputs
        for i in range(self.sysd.states):
            plt.subplot(subplot_max, 1, i + 1)
            if self.sysd.states + self.sysd.inputs > 3:
                plt.ylabel(
                    self.state_labels[i],
                    horizontalalignment="right",
                    verticalalignment="center",
                    rotation=45,
                )
            else:
                plt.ylabel(self.state_labels[i])
            if i == 0:
                if title == self.__default:
                    plt.title("Time-domain responses")
                else:
                    plt.title(title)
            plt.plot(t, self.extract_row(x_rec, i), label="Estimated state")
            plt.plot(t, self.extract_row(ref_rec, i), label="Reference")
            plt.legend()

        for i in range(self.sysd.inputs):
            plt.subplot(subplot_max, 1, self.sysd.states + i + 1)
            if self.sysd.states + self.sysd.inputs > 3:
                plt.ylabel(
                    self.u_labels[i],
                    horizontalalignment="right",
                    verticalalignment="center",
                    rotation=45,
                )
            else:
                plt.ylabel(self.u_labels[i])
            plt.plot(t, self.extract_row(u_rec, i), label="Control effort")
            plt.legend()
        plt.xlabel("Time (s)")

    def set_plot_labels(self, state_labels, u_labels):
        """Sets label data for time-domain response plots.

        Keyword arguments:
        state_labels -- list of tuples containing name of state and the unit.
        u_labels -- list of tuples containing name of input and the unit.
        """
        self.state_labels = [x[0] + " (" + x[1] + ")" for x in state_labels]
        self.u_labels = [x[0] + " (" + x[1] + ")" for x in u_labels]

    def __make_cost_matrix(self, elems):
        """Creates a cost matrix from the given vector for use with LQR.

        The cost matrix is constructed using Bryson's rule. The inverse square
        of each element in the input is taken and placed on the cost matrix
        diagonal.

        Keyword arguments:
        elems -- a vector. For a Q matrix, its elements are the maximum allowed
                 excursions of the states from the reference. For an R matrix,
                 its elements are the maximum allowed excursions of the control
                 inputs from no actuation.

        Returns:
        State excursion or control effort cost matrix
        """
        return np.diag(1.0 / np.square(elems))

    def __make_cov_matrix(self, elems):
        """Creates a covariance matrix from the given vector for use with Kalman
        filters.

        Each element is squared and placed on the covariance matrix diagonal.

        Keyword arguments:
        elems -- a vector. For a Q matrix, its elements are the standard
                 deviations of each state from how the model behaves. For an R
                 matrix, its elements are the standard deviations for each
                 output measurement.

        Returns:
        Process noise or measurement noise covariance matrix
        """
        return np.diag(np.square(elems))

    def export_cpp_coeffs(
        self,
        class_name,
        header_path_prefix="",
        header_extension="hpp",
        period_variant=False,
    ):
        """Exports matrices to pair of C++ source files.

        Keyword arguments:
        class_name -- subsystem class name in camel case
        header_path_prefix -- path prefix in which header exists
        header_extension -- file extension of header file (default: "hpp")
        period_variant -- True to use PeriodVariantLoop, False to use
                          StateSpaceLoop
        """
        from . import system_writer

        system_writer = system_writer.SystemWriter(
            self, class_name, header_path_prefix, header_extension, period_variant
        )
        system_writer.write_cpp_header()
        system_writer.write_cpp_source()

    def export_java_coeffs(self, class_name, period_variant=False):
        """Exports matrices to a Java source file.

        Keyword arguments:
        class_name -- subsystem class name in camel case
        period_variant -- True to use PeriodVariantLoop, False to use
                          StateSpaceLoop
        """
        system_writer = fct.system_writer.SystemWriter(
            self, class_name, "", "", period_variant
        )
        system_writer.write_java_source()
