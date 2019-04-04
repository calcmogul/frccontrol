def runge_kutta(f, x, u, dt):
    """Fourth order Runge-Kutta integration.

    Keyword arguments:
    f -- vector function to integrate
    x -- vector of states
    u -- vector of inputs (constant for dt)
    dt -- time for which to integrate
    """
    half_dt = dt * 0.5
    k1 = f(x, u)
    k2 = f(x + half_dt * k1, u)
    k3 = f(x + half_dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
