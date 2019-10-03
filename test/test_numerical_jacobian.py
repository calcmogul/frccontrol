import numpy as np

from frccontrol.numerical_jacobian import *


def test_numerical_jacobian():
    A = np.array([[1, 2, 4, 1], [5, 2, 3, 4], [5, 1, 3, 2], [1, 1, 3, 7]])
    B = np.array([[1, 1], [2, 1], [3, 2], [3, 7]])

    def AxBuFn(x, u):
        return A @ x + B @ u

    newA = numerical_jacobian_x(4, 4, AxBuFn, np.zeros((4, 1)), np.zeros((2, 1)))
    assert np.allclose(A, newA)

    newB = numerical_jacobian_u(4, 2, AxBuFn, np.zeros((4, 1)), np.zeros((2, 1)))
    assert np.allclose(B, newB)
