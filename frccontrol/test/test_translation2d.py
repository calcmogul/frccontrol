"""Tests the Translation2d class."""

import math

import pytest
from frccontrol.geometry import Rotation2d, Translation2d


def test_sum():
    one = Translation2d(1, 3)
    two = Translation2d(2, 5)

    sum = one + two

    assert sum.x == 3.0
    assert sum.y == 8.0


def test_difference():
    one = Translation2d(1, 3)
    two = Translation2d(2, 5)

    difference = one - two

    assert difference.x == -1.0
    assert difference.y == -2.0


def test_rotate_by():
    another = Translation2d(3, 0)
    rotated = another.rotate_by(Rotation2d.from_degrees(90))

    assert rotated.x == pytest.approx(0.0, abs=1e-9)
    assert rotated.y == pytest.approx(3.0, abs=1e-9)


def test_rotate_around():
    translation = Translation2d(2, 1)
    other = Translation2d(3, 2)
    rotated = translation.rotate_around(other, Rotation2d.from_degrees(180))

    assert rotated.x == pytest.approx(4.0, abs=1e-9)
    assert rotated.y == pytest.approx(3.0, abs=1e-9)


def test_multiplication():
    original = Translation2d(3, 5)
    mult = original * 3

    assert mult.x == 9.0
    assert mult.y == 15.0


def test_division():
    original = Translation2d(3, 5)
    div = original / 2

    assert div.x == 1.5
    assert div.y == 2.5


def test_norm():
    one = Translation2d(3, 5)
    assert one.norm() == math.hypot(3.0, 5.0)


def test_squared_norm():
    one = Translation2d(3, 5)
    assert one.squared_norm() == 34.0


def test_distance():
    one = Translation2d(1, 1)
    two = Translation2d(6, 6)
    assert one.distance(two) == 5.0 * math.sqrt(2.0)


def test_squared_distance():
    one = Translation2d(1, 1)
    two = Translation2d(6, 6)
    assert one.squared_distance(two) == 50.0


def test_unary_minus():
    original = Translation2d(-4.5, 7)
    inverted = -original

    assert inverted.x == 4.5
    assert inverted.y == -7.0


def test_equality():
    one = Translation2d(9, 5.5)
    two = Translation2d(9, 5.5)
    assert one == two


def test_inequality():
    one = Translation2d(9, 5.5)
    two = Translation2d(9, 5.7)
    assert one != two


def test_polar_constructor():
    one = Translation2d.from_polar(math.sqrt(2), Rotation2d.from_degrees(45))
    assert one.x == pytest.approx(1.0, abs=1e-9)
    assert one.y == 1.0

    two = Translation2d.from_polar(2, Rotation2d.from_degrees(60))
    assert two.x == pytest.approx(1.0, abs=1e-9)
    assert two.y == math.sqrt(3.0)


def test_dot():
    one = Translation2d(2, 3)
    two = Translation2d(3, 4)
    assert one.dot(two) == 18.0


def test_cross():
    one = Translation2d(2, 3)
    two = Translation2d(3, 4)
    assert one.cross(two) == -1.0
