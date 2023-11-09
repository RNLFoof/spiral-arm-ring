import pytest

from main import angle_on_which_side_of_line, Side, HorizontalLineException, ParallelLineException


@pytest.mark.parametrize("params", [
    (0  , 0, HorizontalLineException),
    (0.5, 0, HorizontalLineException),
    (1  , 0, HorizontalLineException),
    (1.5, 0, HorizontalLineException),
    (2  , 0, HorizontalLineException),
    (2.5, 0, HorizontalLineException),
    (3  , 0, HorizontalLineException),
    (3.5, 0, HorizontalLineException),

    (0  , 1, Side.RIGHT),
    (0.5, 1, Side.RIGHT),
    (1  , 1, ParallelLineException),
    (1.5, 1, Side.LEFT),
    (2  , 1, Side.LEFT),
    (2.5, 1, Side.LEFT),
    (3  , 1, ParallelLineException),
    (3.5, 1, Side.RIGHT),

    (45/360*4, 1, Side.RIGHT),
    (135/360*4, 1, Side.LEFT),
    ((180+45)/360*4, 1, Side.LEFT),
    ((270+45)/360*4, 1, Side.RIGHT),

    (0  , 1.5, Side.RIGHT),
    (0.5, 1.5, Side.RIGHT),
    (1  , 1.5, Side.RIGHT),
    (1.5, 1.5, ParallelLineException),
    (2  , 1.5, Side.LEFT),
    (2.5, 1.5, Side.LEFT),
    (3  , 1.5, Side.LEFT),
    (3.5, 1.5, ParallelLineException),

    (0  , 2, HorizontalLineException),
    (0.5, 2, HorizontalLineException),
    (1  , 2, HorizontalLineException),
    (1.5, 2, HorizontalLineException),
    (2  , 2, HorizontalLineException),
    (2.5, 2, HorizontalLineException),
    (3  , 2, HorizontalLineException),
    (3.5, 2, HorizontalLineException),

    (0  , 3.5, Side.RIGHT),
    (0.5, 3.5, Side.RIGHT),
    (1  , 3.5, Side.RIGHT),
    (1.5, 3.5, ParallelLineException),
    (2  , 3.5, Side.LEFT),
    (2.5, 3.5, Side.LEFT),
    (3  , 3.5, Side.LEFT),
    (3.5, 3.5, ParallelLineException),
])
def test_angle_on_which_side_of_line(params):
    testee_angle, line_angle, expectation = params
    if expectation in (HorizontalLineException, ParallelLineException):
        with pytest.raises(expectation):
            angle_on_which_side_of_line(testee_angle, line_angle)
    else:
        assert angle_on_which_side_of_line(testee_angle, line_angle) == expectation