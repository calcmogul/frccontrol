"""Tests the Pose2d class."""

import math

import pytest
from frccontrol.geometry import Pose2d, Rotation2d, Transform2d, Translation2d


def test_rotate_by():
    x = 1
    y = 2
    initial = Pose2d.from_triplet(x, y, math.radians(45))

    rotation = Rotation2d.from_degrees(5)
    rotated = initial.rotate_by(rotation)

    # Translation is rotated by CCW rotation matrix
    c = rotation.cos
    s = rotation.sin
    assert rotated.x == c * x - s * y
    assert rotated.y == s * x + c * y
    assert rotated.rotation.degrees == pytest.approx(
        initial.rotation.degrees + rotation.degrees, abs=1e-9
    )


def test_transform_by():
    initial = Pose2d.from_triplet(1, 2, math.radians(45))
    transform = Transform2d(Translation2d(5, 0), Rotation2d.from_degrees(5))

    transformed = initial + transform

    assert transformed.x == 1 + 5 / math.sqrt(2)
    assert transformed.y == 2 + 5 / math.sqrt(2)
    assert transformed.rotation.degrees == pytest.approx(50.0, abs=1e-9)


def test_relative_to():
    initial = Pose2d.from_triplet(0, 0, math.radians(45))
    final = Pose2d.from_triplet(5, 5, math.radians(45))

    final_relative_to_initial = final.relative_to(initial)

    assert final_relative_to_initial.x == pytest.approx(5.0 * math.sqrt(2.0), abs=1e-9)
    assert final_relative_to_initial.y == pytest.approx(0.0, abs=1e-9)
    assert final_relative_to_initial.rotation.degrees == pytest.approx(0.0, abs=1e-9)


def test_rotate_around():
    initial = Pose2d.from_triplet(5, 0, 0)
    point = Translation2d(0, 0)

    rotated = initial.rotate_around(point, Rotation2d.from_degrees(180))

    assert rotated.x == pytest.approx(-5.0, abs=1e-9)
    assert rotated.y == pytest.approx(0.0, abs=1e-9)
    assert rotated.rotation.degrees == pytest.approx(180.0, abs=1e-9)


def test_equality():
    a = Pose2d.from_triplet(0, 5, math.radians(43))
    b = Pose2d.from_triplet(0, 5, math.radians(43))
    assert a == b


def test_inequality():
    a = Pose2d.from_triplet(0, 5, math.radians(43))
    b = Pose2d.from_triplet(0, 1.524, math.radians(43))
    assert a != b


def test_minus():
    initial = Pose2d.from_triplet(0, 0, math.radians(45))
    final = Pose2d.from_triplet(5, 5, math.radians(45))

    transform = final - initial

    assert transform.x == pytest.approx(5.0 * math.sqrt(2.0), abs=1e-9)
    assert transform.y == pytest.approx(0.0, abs=1e-9)
    assert transform.rotation.degrees == pytest.approx(0.0, abs=1e-9)
