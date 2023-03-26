"""
Cubic spline trajectory generator.
"""

import numpy as np


class Trajectory:
    """
    Cubic spline trajectory.
    """

    def __init__(self, times, states):
        """
        Initialize a Trajectory.

        Keyword arguments:
        times -- 1D array of trajectory timestamps.
        states -- 2D array of corresponding states, where each state is a column
        """
        self.times = times.flatten()
        if states.shape[1] != len(times):
            raise RuntimeError(
                f"must have same number of times and states; {len(times)} != {states.shape[1]}"
            )
        self.states = states
        self.start_time = times[0]
        self.end_time = times[-1]

    def clip_time(self, time):
        """
        Limit Trajectory timestamp between start_time and end_time.

        Keyword arguments:
        time -- the time to clip
        """
        return np.clip(time, self.start_time, self.end_time)

    def insert(self, time, state):
        """
        Insert state at the given time.

        Keyword arguments:
        time -- the time
        state -- the state
        """
        if state.ndim == 1:
            state = np.array([state]).T

        if state.shape[0] != self.states.shape[0]:
            raise RuntimeError(
                f"state must have the same number of rows; {state.shape[0]} != {self.states.shape[0]}"
            )

        if state.shape[1] != 1:
            raise RuntimeError(
                f"state must have exactly one column, had {state.shape[1]}"
            )

        before_idx_list = np.where(self.times <= time)[0]
        after_idx_list = np.where(self.times >= time)[0]

        # if there are no elements before the new time
        if before_idx_list.size == 0:
            # add to start of Trajectory
            self.times = np.insert(self.times, 0, time)
            self.states = np.insert(self.states, 0, state.T, axis=1)
            self.start_time = time
            return self
        # if there are no elements after the new time
        elif after_idx_list.size == 0:
            # add to end of Trajectory
            self.times = np.append(self.times, time)
            self.states = np.append(self.states, state, axis=1)
            self.end_time = time
            return self

        prev_idx = before_idx_list[-1]
        next_idx = after_idx_list[0]

        if self.times[prev_idx] == time:
            # time already in Trajectory; overwrite
            self.states[:, prev_idx] = state.flat
        elif self.times[next_idx] == time:
            # time already in Trajectory; overwrite
            self.states[:, next_idx] = state.flat
        else:
            # add to middle of Trajectory
            self.times = np.insert(self.times, next_idx, time)
            self.states = np.insert(self.states, next_idx, state.T, axis=1)

        return self

    def sample(self, time):
        """
        Sample the trajectory for the given time.

        Linearly interpolates between trajectory samples. If time is outside of
        trajectory, gives the start/end state.

        Keyword arguments:
        time -- time to sample
        """
        time = self.clip_time(time)
        prev_idx = np.where(self.times <= time)[0][-1]
        next_idx = np.where(self.times >= time)[0][0]

        if prev_idx == next_idx:
            return np.array([self.states[:, prev_idx]]).T

        prev_val = np.array([self.states[:, prev_idx]]).T
        next_val = np.array([self.states[:, next_idx]]).T
        prev_time = self.times[prev_idx]
        next_time = self.times[next_idx]

        return prev_val + (next_val - prev_val) / (next_time - prev_time) * (
            time - prev_time
        )

    def append(self, other):
        """
        Append another trajectory to this trajectory.

        Will adjust timestamps on the appended trajectory so it starts
        immediately after the current trajectory ends. Skips the first element
        of the other trajectory to avoid repeats.

        Keyword arguments:
        other -- The other trajectory to append to this one.
        """
        # Create new trajectory based off of this one
        combined = Trajectory(self.times, self.states)

        # Adjust timestamps on other trajectory
        other.times = other.times + combined.end_time - other.start_time

        # Combine the time and states
        combined.times = np.concatenate((combined.times, other.times[1:]))
        combined.states = np.concatenate((combined.states, other.states[:, 1:]), 1)

        # Update the end time
        combined.end_time = max(combined.times)

        return combined


def from_coeffs(coeffs: np.matrix, t_0, t_f, n=100) -> Trajectory:
    """
    Generate a trajectory from a polynomial coefficients matrix.

    Keyword arguments:
    coeffs -- Polynomial coefficients as columns in increasing order.
              Can have arbitrarily many columns.
    t_0 -- time to start the interpolation
    t_f -- time to end the interpolation
    n -- number of interpolation samples (default 100)

    Returns:
    Trajectory following the interpolation. The states will be in the form:
    [pos_1, ..., pos_n, vel_1, ..., vel_n, accel_1, ..., accel_n]
    Where n is the number of columns in coeffs.
    """
    order = np.size(coeffs, 0) - 1
    t = np.array([np.linspace(t_0, t_f, n)]).T

    pos_t_vec = np.power(t, np.arange(order + 1))
    pos_vec = pos_t_vec @ coeffs

    vel_t_vec = np.concatenate(
        (
            np.zeros((n, 1)),
            np.multiply(
                pos_t_vec[:, 0:-1], np.repeat(np.array([np.arange(order) + 1]), n, 0)
            ),
        ),
        1,
    )
    vel_vec = vel_t_vec @ coeffs

    acc_t_vec = np.concatenate(
        (
            np.zeros((n, 2)),
            np.multiply(
                vel_t_vec[:, 1:-1],
                np.repeat(np.array([np.arange(order - 1) + 2]), n, 0),
            ),
        ),
        1,
    )
    acc_vec = acc_t_vec @ coeffs

    states = np.concatenate((pos_vec, vel_vec, acc_vec), 1).T
    return Trajectory(t, states)


def interpolate_states(t_0, t_f, state_0, state_f):
    """
    Fit cubic polynomial between states.

    Keyword arguments:
    t_0 -- initial time
    t_f -- final time
    state_0 -- initial state
    state_f -- final state
    """
    return from_coeffs(__cubic_interpolation(t_0, t_f, state_0, state_f), t_0, t_f)


def __cubic_interpolation(t_0, t_f, state_0, state_f):
    """
    Perform cubic interpolation between state₀ at t = t₀ and state_f at t = t_f.
    Solves using the matrix equation:

      [1  t_0   t_0²   t_0³][c₀₁  c₀₂]   [x_0₁  x_0₂]
      [0  1    2t_0   3t_0²][c₁₁  c₁₂] = [v_0₁  v_0₂]
      [1  t_f   t_f²   t_f³][c₂₁  c₂₂]   [x_f₁  x_f₂]
      [0  1    2t_f   3t_f²][c₃₁  c₃₂]   [v_f₁  v_f₂]

    To find the cubic polynomials:

      x₁(t) = c₀₁ + c₁₁t + c₂₁t² + c₃₁t³
      x₂(t) = c₀₂ + c₁₂t + c₂₂t² + c₃₂t³

    where x₁ is the first joint position and x₂ is the second joint position,
    such that the arm is in state_0 [x_0₁, x_0₂, v_0₁, v_0₂]ᵀ at t_0 and state_f
    [x_f₁, x_f₂, v_f₁, v_f₂]ᵀ at t_f.

    Make sure to only use the interpolated cubic for t between t_0 and t_f.

    Keyword arguments:
    t_0 -- start time of interpolation
    t_f -- end time of interpolation
    state_0 -- start state [θ₁, θ₂, ω₁, ω₂]ᵀ
    state_f -- end state [θ₁, θ₂, ω₁, ω₂]ᵀ

    Returns:
    coeffs -- 4x2 matrix containing the interpolation coefficients for joint 1
              in column 1 and joint 2 in column 2
    """

    def pos_row(t):
        return np.array([[1, t, t * t, t * t * t]])

    def vel_row(t):
        return np.array([[0, 1, 2 * t, 3 * t * t]])

    lhs = np.concatenate((pos_row(t_0), vel_row(t_0), pos_row(t_f), vel_row(t_f)))
    rhs = np.concatenate((state_0.reshape((2, 2)), state_f.reshape(2, 2)))

    return np.linalg.inv(lhs) @ rhs
