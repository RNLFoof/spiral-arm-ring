from typing import NamedTuple

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
# It would make more sense to use ASCII stuff here, but this is more fun
from numpy import pi as 𝜋

import main
from main import angle_on_which_side_of_line, Side, HorizontalLineException, ParallelLineException, 𝑒𝑖, PointBundle, \
    PointAndMetadataThereof, CENTER, CORNER, LEFT_EDGE, RIGHT_EDGE

𝜏 = 2 * 𝜋
𝑖 = 1j

@pytest.mark.parametrize("params", [
    (0/8 * 𝜏, 0, HorizontalLineException),
    (1/8 * 𝜏, 0, HorizontalLineException),
    (2/8 * 𝜏, 0, HorizontalLineException),
    (3/8 * 𝜏, 0, HorizontalLineException),
    (4/8 * 𝜏, 0, HorizontalLineException),
    (5/8 * 𝜏, 0, HorizontalLineException),
    (6/8 * 𝜏, 0, HorizontalLineException),
    (7/8 * 𝜏, 0, HorizontalLineException),

    (0/8 * 𝜏, 2/8 * 𝜏, Side.RIGHT),
    (1/8 * 𝜏, 2/8 * 𝜏, Side.RIGHT),
    (2/8 * 𝜏, 2/8 * 𝜏, ParallelLineException),
    (3/8 * 𝜏, 2/8 * 𝜏, Side.LEFT),
    (4/8 * 𝜏, 2/8 * 𝜏, Side.LEFT),
    (5/8 * 𝜏, 2/8 * 𝜏, Side.LEFT),
    (6/8 * 𝜏, 2/8 * 𝜏, ParallelLineException),
    (7/8 * 𝜏, 2/8 * 𝜏, Side.RIGHT),

    (0/8 * 𝜏, 3/8 * 𝜏, Side.RIGHT),
    (1/8 * 𝜏, 3/8 * 𝜏, Side.RIGHT),
    (2/8 * 𝜏, 3/8 * 𝜏, Side.RIGHT),
    (3/8 * 𝜏, 3/8 * 𝜏, ParallelLineException),
    (4/8 * 𝜏, 3/8 * 𝜏, Side.LEFT),
    (5/8 * 𝜏, 3/8 * 𝜏, Side.LEFT),
    (6/8 * 𝜏, 3/8 * 𝜏, Side.LEFT),
    (7/8 * 𝜏, 3/8 * 𝜏, ParallelLineException),

    (0/8 * 𝜏, 4/8 * 𝜏, HorizontalLineException),
    (1/8 * 𝜏, 4/8 * 𝜏, HorizontalLineException),
    (2/8 * 𝜏, 4/8 * 𝜏, HorizontalLineException),
    (3/8 * 𝜏, 4/8 * 𝜏, HorizontalLineException),
    (4/8 * 𝜏, 4/8 * 𝜏, HorizontalLineException),
    (5/8 * 𝜏, 4/8 * 𝜏, HorizontalLineException),
    (6/8 * 𝜏, 4/8 * 𝜏, HorizontalLineException),
    (7/8 * 𝜏, 4/8 * 𝜏, HorizontalLineException),

    (0/8 * 𝜏, 7/8 * 𝜏, Side.RIGHT),
    (1/8 * 𝜏, 7/8 * 𝜏, Side.RIGHT),
    (2/8 * 𝜏, 7/8 * 𝜏, Side.RIGHT),
    (3/8 * 𝜏, 7/8 * 𝜏, ParallelLineException),
    (4/8 * 𝜏, 7/8 * 𝜏, Side.LEFT),
    (5/8 * 𝜏, 7/8 * 𝜏, Side.LEFT),
    (6/8 * 𝜏, 7/8 * 𝜏, Side.LEFT),
    (7/8 * 𝜏, 7/8 * 𝜏, ParallelLineException),
])

def test_angle_on_which_side_of_line(params):
    testee_angle, line_angle, expectation = params
    if expectation in (HorizontalLineException, ParallelLineException):
        with pytest.raises(expectation):
            angle_on_which_side_of_line(testee_angle, line_angle)
    else:
        assert angle_on_which_side_of_line(testee_angle, line_angle) == pytest.approx(expectation)

@pytest.mark.parametrize("fraction", np.linspace(0, 1, 8, endpoint=False))
def test_ei(fraction):
   assert 𝑒𝑖(fraction * 𝜏) == pytest.approx(1j ** (fraction * 4))


class TestPointAndMetadataThereof:
    @staticmethod
    def test_getitem():
        guy = PointAndMetadataThereof((2, 3), ring=4)
        assert guy[0] == 2
        assert guy[1] == 3


class TestPointBundle:
    class TestCircleIntersectionExpectations(NamedTuple):
        center: tuple[float, float]
        corner: tuple[float, float]
        pixel: tuple[float, float]
        intersection: tuple[float, float] | None

    @staticmethod
    @pytest.mark.parametrize("expectations", [
        TestCircleIntersectionExpectations((0, 0), (1, 0), (1.5, 0), (2, 0)),
        TestCircleIntersectionExpectations((0, 0), (1, 0), (5, 0), None),
        TestCircleIntersectionExpectations((0, 0), (1, 0), (2, 0), (2, 0)),
        TestCircleIntersectionExpectations((0, 1), (1, 1), (2, 1), (2, 1)),
        TestCircleIntersectionExpectations((0, 1), (1, 1), (2, 1), (2, 1)),
        TestCircleIntersectionExpectations((0, 0), (0, 0), (0, 0), (2, 0)),
        TestCircleIntersectionExpectations((0, 0), (1, 0), (0, 0), (-2, 0)),
        TestCircleIntersectionExpectations((0, 0), (0, 0), (1, 0), (2, 0)),
        TestCircleIntersectionExpectations((0, 0), (0, 2), (0, 0), (0, -2)),
        TestCircleIntersectionExpectations((0, 0), (0, 2), (0, -2), (0, -2)),
    ])
    def test_circle_intersection(monkeypatch: MonkeyPatch, expectations: TestCircleIntersectionExpectations):
        ring = 4  # Ring()
        bundle = PointBundle({
            CENTER: PointAndMetadataThereof(expectations.center, ring=ring),
            CORNER: PointAndMetadataThereof(expectations.corner, ring=ring),
            LEFT_EDGE: PointAndMetadataThereof((0, 0), ring=ring),
            RIGHT_EDGE: PointAndMetadataThereof((0, 0), ring=ring),
        })
        monkeypatch.setattr(main.PointBundle, "current_arm_radius", 2)
        assert bundle.circle_intersection(expectations.pixel) == expectations.intersection

    class TestCornerColorExpectations(NamedTuple):
        center: tuple[float, float]
        corner: tuple[float, float]
        pixel: tuple[float, float]
        color: float | None

    @staticmethod
    @pytest.mark.parametrize("expectations", [
        TestCornerColorExpectations((0, 0), (0, 0), (0, 0), 0),
        TestCornerColorExpectations((0, 0), (0, 0), (2, 0), 255),
        TestCornerColorExpectations((0, 0), (0, 0), (0, 2), 255),
        TestCornerColorExpectations((0, 0), (0, 2), (0, 2), 0),
        TestCornerColorExpectations((0, 0), (0, 2), (0, 1), 64),
        TestCornerColorExpectations((0, 0), (0, 1), (0, 2), 255),
        TestCornerColorExpectations((0, 0), (0, 2), (0, 0), 128),
        TestCornerColorExpectations((0, 0), (0, 2), (0, -2), 255),
    ])
    def test_corner_color(monkeypatch: MonkeyPatch, expectations: TestCornerColorExpectations):
        ring = 4  # Ring()
        bundle = PointBundle({
            CENTER: PointAndMetadataThereof(expectations.center, ring=ring),
            CORNER: PointAndMetadataThereof(expectations.corner, ring=ring),
            LEFT_EDGE: PointAndMetadataThereof((0, 0), ring=ring),
            RIGHT_EDGE: PointAndMetadataThereof((0, 0), ring=ring),
        })
        monkeypatch.setattr(main.PointBundle, "current_arm_radius", 2)
        assert bundle.corner_color(expectations.pixel) == expectations.color
