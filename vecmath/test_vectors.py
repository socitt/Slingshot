import math

import pytest

from vecmath.vectors import Vec2, add, axpy, dot, lincomb, max_abs_diff, norm, scale, sub


def test_vec2_arithmetic():
    a, b = Vec2(1, 2), Vec2(3, -4)
    assert a + b == Vec2(4, -2)
    assert a - b == Vec2(-2, 6)
    assert 2 * a == a * 2 == Vec2(2, 4)
    assert -a == Vec2(-1, -2)
    assert b / 2 == Vec2(1.5, -2)


def test_vec2_geometry():
    b = Vec2(3, -4)
    assert b.norm() == 5.0
    assert b.norm2() == 25.0
    assert Vec2(1, 0).dot(Vec2(0, 1)) == 0.0
    assert Vec2(1, 0).cross(Vec2(0, 1)) == 1.0
    assert Vec2(1, 0).perp() == Vec2(0, 1)
    assert Vec2(1, 0).rotated(math.pi / 2).is_close(Vec2(0, 1))
    assert Vec2.polar(2, math.pi).is_close(Vec2(-2, 0))
    assert math.isclose(b.normalized().norm(), 1.0)


def test_vec2_is_immutable_and_hashable():
    v = Vec2(1, 2)
    with pytest.raises(AttributeError):
        v.x = 5
    assert {v: 1}[Vec2(1.0, 2.0)] == 1
    assert list(v) == [1.0, 2.0]


def test_zero_normalize_raises():
    with pytest.raises(ZeroDivisionError):
        Vec2().normalized()


def test_flat_helpers():
    a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    assert add(a, b) == [5.0, 7.0, 9.0]
    assert sub(b, a) == [3.0, 3.0, 3.0]
    assert scale(a, 2) == [2.0, 4.0, 6.0]
    assert axpy(2.0, a, b) == [6.0, 9.0, 12.0]
    assert lincomb(a, [(1.0, b), (-1.0, a)]) == b
    assert dot(a, b) == 32.0
    assert norm([3.0, 4.0]) == 5.0
    assert max_abs_diff(a, [1.0, 2.5, 3.0]) == 0.5


def test_lincomb_does_not_mutate_base():
    base = [0.0, 0.0]
    lincomb(base, [(1.0, [1.0, 1.0])])
    assert base == [0.0, 0.0]


def test_max_abs_diff_length_mismatch():
    with pytest.raises(ValueError):
        max_abs_diff([1.0], [1.0, 2.0])
