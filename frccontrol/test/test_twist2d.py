"""Tests the Twist2d class."""

import math

import pytest
from frccontrol.geometry import Pose2d, Twist2d


def test_straight():
    straight = Twist2d(5, 0, 0)
    straight_transform = straight.exp()

    assert straight_transform.x == 5.0
    assert straight_transform.y == 0.0
    assert straight_transform.rotation.radians == 0.0


def test_quarter_circle():
    quarter_circle = Twist2d(5 / 2 * math.pi, 0, math.pi / 2)
    quarter_circle_transform = quarter_circle.exp()

    assert quarter_circle_transform.x == 5.0
    assert quarter_circle_transform.y == pytest.approx(5.0, abs=1e-9)
    assert quarter_circle_transform.rotation.degrees == 90.0


def test_diagonal_no_dtheta():
    diagonal = Twist2d(2, 2, 0)
    diagonal_transform = diagonal.exp()

    assert diagonal_transform.x == 2.0
    assert diagonal_transform.y == 2.0
    assert diagonal_transform.rotation.degrees == 0.0


def test_equality():
    one = Twist2d(5, 1, 3)
    two = Twist2d(5, 1, 3)
    assert one == two


def test_inequality():
    one = Twist2d(5, 1, 3)
    two = Twist2d(5, 1.2, 3)
    assert one != two


def test_pose2d_log():
    end = Pose2d.from_triplet(5, 5, math.radians(90))
    start = Pose2d.from_triplet(0, 0, 0)

    twist = (end - start).log()

    assert twist == Twist2d(5 / 2 * math.pi, 0, math.pi / 2)

    # Make sure computed twist gives back original end pose
    reapplied = start + twist.exp()
    assert reapplied == end
