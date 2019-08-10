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


def pzmap(sys, title):
    """Plot poles and zeroes of discrete system.

    Keyword arguments:
    sys -- the system to plot
    title -- the title of the plot
    """
    poles = sys.pole()
    zeros = sys.zero()

    if sys.dt != None:
        ax, fig = ct.grid.zgrid()
    else:
        ax, fig = ct.grid.sgrid()

    # Plot the locations of the poles and zeros
    if len(poles) > 0:
        ax.scatter(np.real(poles), np.imag(poles), s=50, marker="x", facecolors="k")
    if len(zeros) > 0:
        ax.scatter(
            np.real(zeros),
            np.imag(zeros),
            s=50,
            marker="o",
            facecolors="none",
            edgecolors="k",
        )

    plt.title(title)

    if sys.dt != None:
        circle = plt.Circle((0, 0), radius=1, fill=False)
        ax.add_artist(circle)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))


def plot_open_loop_poles(system, discrete=True):
    """Plot open-loop poles.

    Keyword arguments:
    system -- a System instance
    discrete -- whether to make pole-zero map of continuous or discrete version
                of system
    """
    if discrete:
        ss = system.sysd
    else:
        ss = system.sysc

    print("Open-loop poles =", ss.pole())
    print("Open-loop zeroes =", ss.zero())
    pzmap(ss, title="Open-loop pole-zero map")


def plot_closed_loop_poles(system, discrete=True):
    """Plot closed-loop poles.

    Keyword arguments:
    system -- a System instance
    discrete -- whether to make pole-zero map of continuous or discrete version
                of system
    """
    if discrete:
        ss = system.sysd
    else:
        ss = system.sysc

    ss_cl = ct.StateSpace(
        ss.A - ss.B @ system.K,
        ss.B @ system.K,
        ss.C - ss.D @ system.K,
        ss.D @ system.K,
        ss.dt,
    )
    print("Closed-loop poles =", ss_cl.pole())
    print("Closed-loop zeroes =", ss_cl.zero())
    pzmap(ss_cl, title="Closed-loop pole-zero map")


def plot_observer_poles(system, discrete=True):
    """Plot discrete observer poles.

    Keyword arguments:
    system -- a System instance
    discrete -- whether to make pole-zero map of continuous or discrete version
                of system
    """
    if discrete:
        ss = system.sysd
    else:
        ss = system.sysc

    ss_cl = ct.StateSpace(
        ss.A - ss.A @ system.kalman_gain @ ss.C, ss.B, ss.C, ss.D, ss.dt
    )
    print("Observer poles =", ss_cl.pole())
    pzmap(ss_cl, title="Observer poles")
