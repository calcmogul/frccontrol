"""Control system utility functions."""

import control as ct
import matplotlib.pyplot as plt
import numpy as np


def conv(polynomial, *args):
    """Implementation of MATLAB's conv() function.

    Keyword arguments:
    polynomial -- list of coefficients of first polynomial

    Arguments:
    *args -- more lists of polynomial coefficients
    """
    for arg in args:
        polynomial = np.convolve(polynomial, arg).tolist()
    return polynomial


def dpzmap(sys, title):
    """Plot poles and zeroes of discrete system.

    Keyword arguments:
    sys -- the system to plot
    title -- the title of the plot
    """
    ct.pzmap(sys, title=title)
    circle = plt.Circle((0, 0), radius=1, fill=False)
    ax = plt.gca()
    ax.add_artist(circle)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))


def closed_loop_ctrl(system):
    """Constructs the closed-loop system for a discrete controller.

    Keyword arguments:
    system -- a System instance

    Returns:
    StateSpace instance representing closed-loop controller.
    """
    return ct.StateSpace(
        system.sysd.A - system.sysd.B @ system.K,
        system.sysd.B @ system.K,
        system.sysd.C - system.sysd.D @ system.K,
        system.sysd.D @ system.K,
    )


def plot_observer_poles(system):
    """Plot discrete observer poles.

    Keyword arguments:
    system -- a System instance
    """
    sys_cl = ct.StateSpace(
        system.sysd.A - system.sysd.A @ system.kalman_gain @ system.sysd.C,
        system.sysd.B,
        system.sysd.C,
        system.sysd.D,
    )
    dpzmap(sys_cl, title="Observer poles")
