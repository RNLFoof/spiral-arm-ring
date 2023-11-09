import numpy as np
import pytest

from main import angle_on_which_side_of_line, Side, HorizontalLineException, ParallelLineException, 𝑒𝑖

# It would make more sense to use ASCII stuff here, but this is more fun
from numpy import pi as 𝜋
from numpy import e as 𝑒
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