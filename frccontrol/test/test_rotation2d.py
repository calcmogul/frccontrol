"""Tests the Rotation2d class."""

import math

import pytest
from frccontrol.geometry import Rotation2d


def test_radians_to_degrees():
    rot1 = Rotation2d.from_radians(math.pi / 3)
    rot2 = Rotation2d.from_radians(math.pi / 4)

    assert rot1.degrees == pytest.approx(60.0, abs=1e-9)
    assert rot2.degrees == pytest.approx(45.0, abs=1e-9)


def test_degrees_to_radians():
    rot1 = Rotation2d.from_degrees(45)
    rot2 = Rotation2d.from_degrees(30)

    assert rot1.radians == math.pi / 4
    assert rot2.radians == math.pi / 6


def test_rotate_by_from_zero():
    zero = Rotation2d.from_radians(0)
    rotated = zero + Rotation2d.from_degrees(90)

    assert rotated.radians == math.pi / 2
    assert rotated.degrees == 90.0


def test_rotate_by_non_zero():
    rot = Rotation2d.from_degrees(90)
    rot = rot + Rotation2d.from_degrees(30)

    assert rot.degrees == pytest.approx(120.0, abs=1e-9)


def test_minus():
    rot1 = Rotation2d.from_degrees(70)
    rot2 = Rotation2d.from_degrees(30)

    assert (rot1 - rot2).degrees == pytest.approx(40.0, abs=1e-9)


def test_unary_minus():
    rot = Rotation2d.from_degrees(20)

    assert (-rot).degrees == -20.0


def test_multiply():
    rot = Rotation2d.from_degrees(10)

    assert (rot * 3.0).degrees == pytest.approx(30.0, abs=1e-9)
    assert (rot * 41.0).degrees == pytest.approx(50.0, abs=1e-9)


def test_equality():
    rot1 = Rotation2d.from_degrees(43)
    rot2 = Rotation2d.from_degrees(43)
    assert rot1 == rot2

    rot1 = Rotation2d.from_degrees(-180)
    rot2 = Rotation2d.from_degrees(180)
    assert rot1 == rot2


def test_inequality():
    rot1 = Rotation2d.from_degrees(43)
    rot2 = Rotation2d.from_degrees(43.5)
    assert rot1 != rot2
