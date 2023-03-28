"""
Plotting utilities.
"""

import matplotlib.pyplot as plt
import numpy as np


def generate_time_responses(system, refs):
    """
    Generate time-domain responses of the system and the control inputs.

    Keyword arguments:
    system -- system for which to generate responses
    refs -- list of reference vectors, one for each time

    Returns:
    r_rec -- recording of references
    x_rec -- If system.observer exists, a recording of the state estimates.
             Otherwise, a recording of the true states.
    u_rec -- recording of inputs
    y_rec -- recording of outputs
    """
    r_rec = np.zeros((system.x.shape[0], 0))
    x_rec = np.zeros((system.x.shape[0], 0))
    u_rec = np.zeros((system.u.shape[0], 0))
    y_rec = np.zeros((system.y.shape[0], 0))

    # Run simulation
    for i, r in enumerate(refs):
        if i < len(refs) - 1:
            next_r = refs[i + 1]
        else:
            next_r = r
        system.update(r, next_r)

        # Log states for plotting
        r_rec = np.concatenate((r_rec, r), axis=1)
        if hasattr(system, "observer"):
            x_rec = np.concatenate((x_rec, system.observer.x_hat), axis=1)
        else:
            x_rec = np.concatenate((x_rec, system.x), axis=1)
        u_rec = np.concatenate((u_rec, system.u), axis=1)
        y_rec = np.concatenate((y_rec, system.y), axis=1)

    return r_rec, x_rec, u_rec, y_rec


def plot_time_responses(
    state_labels,
    input_labels,
    t_rec,
    r_rec,
    x_rec,
    u_rec,
    title="Time-domain responses",
):
    """Plots time-domain responses of the system and the control inputs.

    Keyword arguments:
    state_labels -- list of state label strings
    input_labels -- list of input label strings
    t_rec -- list of timesteps corresponding to references
    r_rec -- recording of references from generate_time_responses()
    x_rec -- recording of state estimates from generate_time_responses()
    u_rec -- recording of inputs from generate_time_responses()
    title -- title for time-domain plots (default: "Time-domain responses")
    """
    states = x_rec.shape[0]
    inputs = u_rec.shape[0]

    plt.figure()
    subplot_max = states + inputs
    for i in range(states):
        plt.subplot(subplot_max, 1, i + 1)
        if states + inputs > 3:
            plt.ylabel(
                state_labels[i],
                horizontalalignment="right",
                verticalalignment="center",
                rotation=45,
            )
        else:
            plt.ylabel(state_labels[i])
        if i == 0 and title is not None:
            plt.title(title)
        plt.plot(t_rec, r_rec[i, :], label="Reference")
        plt.plot(t_rec, x_rec[i, :], label="State")
        plt.legend()

    for i in range(inputs):
        plt.subplot(subplot_max, 1, states + i + 1)
        if states + inputs > 3:
            plt.ylabel(
                input_labels[i],
                horizontalalignment="right",
                verticalalignment="center",
                rotation=45,
            )
        else:
            plt.ylabel(input_labels[i])
        plt.plot(t_rec, u_rec[i, :], label="Input")
        plt.legend()
    plt.xlabel("Time (s)")
