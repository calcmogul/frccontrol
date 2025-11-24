"""Tests the Transform2d class."""

import math

from frccontrol.geometry import Pose2d, Transform2d


def test_inverse():
    initial = Pose2d.from_triplet(1, 2, math.radians(45))
    transform = Transform2d.from_triplet(5, 0, math.radians(5))

    transformed = initial + transform
    untransformed = transformed + transform.inverse()

    assert untransformed == initial


def test_composition():
    initial = Pose2d.from_triplet(1, 2, math.radians(45))
    transform1 = Transform2d.from_triplet(5, 0, math.radians(5))
    transform2 = Transform2d.from_triplet(0, 2, math.radians(5))

    transformed_separate = initial + transform1 + transform2
    transformed_combined = initial + (transform1 + transform2)

    assert transformed_separate == transformed_combined
