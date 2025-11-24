"""Geometry utilities."""

from __future__ import annotations

import math


class Translation2d:
    """Represents a translation in 2D space."""

    def __init__(self, x: float, y: float):
        self.__x = x
        self.__y = y

    @classmethod
    def from_polar(cls, distance: float, angle: Rotation2d) -> Translation2d:
        """
        Constructs a Translation2d with the provided distance and angle. This is
        essentially converting from polar coordinates to Cartesian coordinates.

        Parameter ``distance``:
            The distance from the origin to the end of the translation.
        Parameter ``angle``:
            The angle between the x-axis and the translation vector.
        """
        return Translation2d(distance * angle.cos, distance * angle.sin)

    @property
    def x(self) -> float:
        """Returns x component."""
        return self.__x

    @property
    def y(self) -> float:
        """Returns y component."""
        return self.__y

    def distance(self, other: Translation2d) -> float:
        """
        Calculates the distance between two translations in 2D space.

        The distance between translations is defined as √((x₂−x₁)²+(y₂−y₁)²).

        Parameter ``other``:
            The translation to compute the distance to.
        """
        return math.hypot(other.x - self.x, other.y - self.y)

    def squared_distance(self, other: Translation2d) -> float:
        """
        Calculates the square of the distance between two translations in 2D
        space. This is equivalent to squaring the result of
        distance(Translation2d), but avoids computing a square root.

        The square of the distance between translations is defined as
        (x₂−x₁)²+(y₂−y₁)².

        Parameter ``other``:
            The translation to compute the squared distance to.
        """
        return (other.x - self.x) ** 2 + (other.y - self.y) ** 2

    def norm(self) -> float:
        """Returns the norm, or distance from the origin to the translation."""
        return math.hypot(self.x, self.y)

    def squared_norm(self) -> float:
        """
        Returns the squared norm, or squared distance from the origin to the
        translation. This is equivalent to squaring the result of Norm(), but
        avoids computing a square root.
        """
        return self.x**2 + self.y**2

    def angle(self) -> Rotation2d:
        """Returns the angle this translation forms with the positive X axis."""
        return Rotation2d(self.x, self.y)

    def rotate_by(self, other: Rotation2d) -> Translation2d:
        """
        Applies a rotation to the translation in 2D space.

        Parameter ``other``:
            The rotation by which to rotate.
        """
        return Translation2d(
            self.x * other.cos - self.y * other.sin,
            self.x * other.sin + self.y * other.cos,
        )

    def rotate_around(self, other: Translation2d, rot: Rotation2d) -> Translation2d:
        """
        Rotates this translation around another translation in 2D space.

        Parameter ``other``:
            The other translation to rotate around.
        Parameter ``rot``:
            The rotation to rotate the translation by.
        """
        return Translation2d(
            (self.x - other.x) * rot.cos - (self.y - other.y) * rot.sin + other.x,
            (self.x - other.x) * rot.sin + (self.y - other.y) * rot.cos + other.y,
        )

    def dot(self, other: Translation2d) -> float:
        """
        Computes the dot product between this translation and another
        translation in 2D space.

        The dot product between two translations is defined as x₁x₂+y₁y₂.

        Parameter ``other``:
            The translation to compute the dot product with.
        """
        return self.x * other.x + self.y * other.y

    def cross(self, other: Translation2d) -> float:
        """
        Computes the cross product between this translation and another
        translation in 2D space.

        The 2D cross product between two translations is defined as x₁y₂-x₂y₁.

        Parameter ``other``:
            The translation to compute the cross product with.
        """
        return self.x * other.y - self.y * other.x

    def __add__(self, other: Translation2d) -> Translation2d:
        return Translation2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Translation2d) -> Translation2d:
        return Translation2d(self.x - other.x, self.y - other.y)

    def __neg__(self) -> Translation2d:
        return Translation2d(-self.x, -self.y)

    def __mul__(self, scalar: float) -> Translation2d:
        return Translation2d(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> Translation2d:
        return Translation2d(self.x / scalar, self.y / scalar)

    def __eq__(self, other) -> bool:
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9


class Rotation2d:
    """Represents a rotation in 2D space."""

    def __init__(self, x: float, y: float):
        magnitude = math.hypot(x, y)
        if magnitude > 1e-6:
            self.__cos = x / magnitude
            self.__sin = y / magnitude
        else:
            raise RuntimeError("x and y components of Rotation2d are zero")

    @classmethod
    def from_radians(cls, heading: float) -> Rotation2d:
        """
        Constructs a Rotation2d from a heading in radians.

        Parameter ``heading``:
            The heading in radians.
        """
        return cls(math.cos(heading), math.sin(heading))

    @classmethod
    def from_degrees(cls, heading: float) -> Rotation2d:
        """
        Constructs a Rotation2d from a heading in degrees.

        Parameter ``heading``:
            The heading in degrees.
        """
        return cls(math.cos(math.radians(heading)), math.sin(math.radians(heading)))

    @property
    def radians(self) -> float:
        """Returns the heading in radians."""
        return math.atan2(self.sin, self.cos)

    @property
    def degrees(self) -> float:
        """Returns the heading in degrees."""
        return math.degrees(math.atan2(self.sin, self.cos))

    @property
    def cos(self) -> float:
        """Returns the cosine of the heading."""
        return self.__cos

    @property
    def sin(self) -> float:
        """Returns the sine of the heading."""
        return self.__sin

    @property
    def tan(self) -> float:
        """Returns the tangent of the heading."""
        return self.sin / self.cos

    def __add__(self, other: Rotation2d) -> Rotation2d:
        return self.rotate_by(other)

    def __sub__(self, other: Rotation2d) -> Rotation2d:
        return self.rotate_by(-other)

    def __neg__(self) -> Rotation2d:
        return Rotation2d(self.cos, -self.sin)

    def rotate_by(self, other: Rotation2d) -> Rotation2d:
        """
        Rotates the rotation in 2D space.

        Parameter ``other``:
            The rotation by which to rotate.
        """
        return Rotation2d(
            self.cos * other.cos - self.sin * other.sin,
            self.cos * other.sin + self.sin * other.cos,
        )

    def __mul__(self, scalar: float) -> Rotation2d:
        return Rotation2d.from_radians(self.radians * scalar)

    def __truediv__(self, scalar: float) -> Rotation2d:
        return Rotation2d.from_radians(self.radians / scalar)

    def __eq__(self, other) -> bool:
        return math.hypot(self.cos - other.cos, self.sin - other.sin) < 1e-9


class Pose2d:
    """Represents a pose in 2D space."""

    def __init__(self, translation: Translation2d, rotation: Rotation2d):
        self.__translation = translation
        self.__rotation = rotation

    @classmethod
    def from_triplet(cls, x: float, y: float, θ: float) -> Pose2d:
        """
        Constructs a Pose2d from an x-y-heading triplet.

        Parameter ``x``:
            The x component.
        Parameter ``y``:
            The y component.
        Parameter ``θ``:
            The heading component.
        """
        return cls(Translation2d(x, y), Rotation2d.from_radians(θ))

    @property
    def translation(self) -> Translation2d:
        """Returns the translation."""
        return self.__translation

    @property
    def rotation(self) -> Rotation2d:
        """Returns the rotation."""
        return self.__rotation

    @property
    def x(self) -> float:
        """Returns the x component."""
        return self.translation.x

    @property
    def y(self) -> float:
        """Returns the y component."""
        return self.translation.y

    def __add__(self, other: Transform2d) -> Pose2d:
        return self.transform_by(other)

    def __sub__(self, other: Pose2d) -> Transform2d:
        pose = self.relative_to(other)
        return Transform2d(pose.translation, pose.rotation)

    def rotate_by(self, other: Rotation2d) -> Pose2d:
        """
        Rotates the pose in 2D space.

        Parameter ``other``:
            The rotation by which to rotate.
        """
        return Pose2d(self.translation.rotate_by(other), self.rotation.rotate_by(other))

    def transform_by(self, other: Transform2d) -> Pose2d:
        """
        Transforms this pose by the given transformation.

        Parameter ``other``:
            The transformation by which to transform.
        """
        return Pose2d(
            self.translation + other.translation.rotate_by(self.rotation),
            other.rotation + self.rotation,
        )

    def relative_to(self, other: Pose2d) -> Pose2d:
        """
        Returns this pose relative to another pose.

        Parameter ``other``:
            The origin for the relative pose.
        """
        transform = Transform2d.from_poses(other, self)
        return Pose2d(transform.translation, transform.rotation)

    def rotate_around(self, point: Translation2d, rot: Rotation2d) -> Pose2d:
        """
        Rotates the pose around a point in 2D space.

        Parameter ``point``:
            The point in 2D space to rotate around.
        Parameter ``rot``:
            The rotation to rotate the pose by.
        """
        return Pose2d(
            self.translation.rotate_around(point, rot), self.rotation.rotate_by(rot)
        )

    def __eq__(self, other) -> bool:
        return (
            self.__translation == other.__translation
            and self.__rotation == other.__rotation
        )


class Transform2d:
    """Represents a transformation of a pose in the pose's frame."""

    def __init__(self, translation: Translation2d, rotation: Rotation2d):
        self.__translation = translation
        self.__rotation = rotation

    @classmethod
    def from_triplet(cls, x: float, y: float, θ: float) -> Transform2d:
        """
        Constructs a Transform2d from an x-y-heading triplet.

        Parameter ``x``:
            The x component.
        Parameter ``y``:
            The y component.
        Parameter ``θ``:
            The heading component.
        """
        return cls(Translation2d(x, y), Rotation2d.from_radians(θ))

    @classmethod
    def from_poses(cls, initial: Pose2d, final: Pose2d) -> Transform2d:
        """
        Constructs a Transform2d from initial and final poses.

        Parameter ``initial``:
            The initial pose.
        Parameter ``final``:
            The final pose.
        """
        return cls(
            (final.translation - initial.translation).rotate_by(-initial.rotation),
            final.rotation - initial.rotation,
        )

    @property
    def translation(self) -> Translation2d:
        """Returns the translation."""
        return self.__translation

    @property
    def rotation(self) -> Rotation2d:
        """Returns the rotation."""
        return self.__rotation

    @property
    def x(self) -> float:
        """Returns the x component."""
        return self.translation.x

    @property
    def y(self) -> float:
        """Returns the y component."""
        return self.translation.y

    def log(self) -> Twist2d:
        """Takes the log of the transformation to obtain a twist."""
        dtheta = self.rotation.radians
        half_dtheta = dtheta / 2

        cos_minus_one = self.rotation.cos - 1

        if abs(cos_minus_one) < 1e-9:
            half_theta_by_tan_of_half_dtheta = 1.0 - 1.0 / 12.0 * dtheta * dtheta
        else:
            half_theta_by_tan_of_half_dtheta = (
                -(half_dtheta * self.rotation.sin) / cos_minus_one
            )

        translation_part = self.translation.rotate_by(
            Rotation2d(half_theta_by_tan_of_half_dtheta, -half_dtheta)
        ) * math.hypot(half_theta_by_tan_of_half_dtheta, half_dtheta)

        return Twist2d(translation_part.x, translation_part.y, dtheta)

    def inverse(self) -> Transform2d:
        """
        Invert the transformation. This is useful for undoing a transformation.
        """
        return Transform2d(
            (-self.translation).rotate_by(-self.rotation), -self.rotation
        )

    def __mul__(self, scalar: float) -> Transform2d:
        return Transform2d(self.translation * scalar, self.rotation * scalar)

    def __truediv__(self, scalar: float) -> Transform2d:
        return self * (1.0 / scalar)

    def __add__(self, other: Transform2d) -> Transform2d:
        """
        Composes two transformations. The second transform is applied relative
        to the orientation of the first.

        Parameter ``other``:
            The transform to compose with this one.
        """
        return Transform2d.from_poses(
            Pose2d.from_triplet(0, 0, 0),
            Pose2d.from_triplet(0, 0, 0).transform_by(self).transform_by(other),
        )

    def __eq__(self, other) -> bool:
        return (
            self.__translation == other.__translation
            and self.__rotation == other.__rotation
        )


class Twist2d:
    """A change in distance in the tangent space of 2D pose."""

    def __init__(self, dx: float, dy: float, dθ: float):
        """
        Constructs a Twist2d.

        Parameter ``dx``:
            Linear "dx" component.
        Parameter ``dy``:
            Linear "dy" component.
        Parameter ``dθ``:
            Angular "dtheta" component (radians).
        """
        self.__dx = dx
        self.__dy = dy
        self.__dθ = dθ

    @property
    def dx(self) -> float:
        """Returns the dx component."""
        return self.__dx

    @property
    def dy(self) -> float:
        """Returns the dy component."""
        return self.__dy

    @property
    def dθ(self) -> float:
        """Returns the dθ component."""
        return self.__dθ

    def exp(self) -> Transform2d:
        """Exponentiates the twist to obtain a transformation."""
        rot = Rotation2d.from_radians(self.dθ)

        if abs(self.dθ) < 1e-9:
            s = 1 - 1 / 6 * self.dθ * self.dθ
            c = 0.5 * self.dθ
        else:
            s = rot.sin / self.dθ
            c = (1 - rot.cos) / self.dθ

        return Transform2d(
            Translation2d(self.dx * s - self.dy * c, self.dx * c + self.dy * s), rot
        )

    def __eq__(self, other) -> bool:
        return (
            abs(self.dx - other.dx) < 1e-9
            and abs(self.dy - other.dy) < 1e-9
            and abs(self.dθ - other.dθ) < 1e-9
        )
