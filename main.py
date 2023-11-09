from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Any

import numpy as np
import scipy
from PIL import Image
from PIL.ImageDraw import ImageDraw
# It would make more sense to use ASCII stuff here, but this is more fun
from numpy import pi as 𝜋
from scipy import interpolate

𝜏 = 2 * 𝜋
𝑖 = 1j

@dataclass
class RingArm:
    starting_radians: float

class Side(Enum):
    LEFT = 0
    RIGHT = 1

class HorizontalLineException(ValueError):
    pass

class ParallelLineException(ValueError):
    pass

def 𝑒𝑖(multiplier: float):
    return np.exp(𝑖 * multiplier)

def angle_on_which_side_of_line(testee_radians: float, line_radians: float):
    line_radians %= 𝜋

    if np.isclose(line_radians, 0):
        raise HorizontalLineException("Dumbass this line is EXACTLY HORIZONTAL!!!! NOTHING is horizontal to it!!!")
    if np.isclose(testee_radians % 𝜋, line_radians):
        raise ParallelLineException("Dumbass this line is EXACTLY THE SAME ANGLE AS THE ANGLE YOU'RE TESTING!!!")

    spin_it = np.imag(𝑒𝑖(testee_radians + (𝜋 - line_radians)))
    if spin_it <= 0:
        return Side.LEFT
    else:
        return Side.RIGHT

@dataclass
class Ring:
    width: int
    height: int
    rotations: float
    outer_multiplier: float
    inner_multiplier: float
    ring_arms: Iterable[RingArm]
    side_dividing_line_radians = 3 / 8 * 𝜏

    def generate(self) -> Image:
        image = Image.new("RGBA", (self.width, self.height))

        for ring_arm in self.ring_arms:
            image = add_an_arm(image, ring_arm, self)

        return image


def add_an_arm(image: Image, arm: RingArm, ring: Ring):
    largest_distance = ring.height * ring.outer_multiplier * 0.5
    width, ring.height = image.size
    draw = ImageDraw(image)

    center_height = ring.height*0.5j
    center = width*0.5 + center_height

    side_of_diagonal = angle_on_which_side_of_line(arm.starting_radians, ring.side_dividing_line_radians)
    side_of_vertical = angle_on_which_side_of_line(arm.starting_radians, 1 / 2 * 𝜋)
    if side_of_diagonal == Side.LEFT:
        start = center_height
    else:
        start = width + center_height

    if side_of_diagonal != side_of_vertical:
        start_from = .25 * 𝜏 if side_of_vertical == Side.LEFT else .75 * 𝜏
    else:
        start_from = arm.starting_radians

    precision = 48
    center_closeness = ring.inner_multiplier
    spiral_complexes = []
    for radians in np.linspace(start_from, arm.starting_radians + ring.rotations * 𝜏, num=precision):
        progress = scipy.interpolate.interp1d(
            [arm.starting_radians, arm.starting_radians + ring.rotations * 𝜏],
            [0, 1],
            fill_value="extrapolate"
        )(radians)
        distance = scipy.interpolate.interp1d(
            [0, 1],
            [largest_distance, largest_distance*center_closeness],
            fill_value="extrapolate"
        )(progress)
        to_append = center + 𝑒𝑖(radians) * distance
        spiral_complexes.append(to_append)

    precision = 48
    buildup_gets_this_close = 0.01
    buildup_complexes = np.interp(
        np.linspace(0, buildup_gets_this_close, num=precision),
        [0, 1],
        [start, spiral_complexes[0]]
    )

    complexes = np.concatenate([buildup_complexes, np.array(spiral_complexes)], axis=0)

    def unpack(xy_complex: complex) -> list[float]:
        return [np.real(xy_complex), np.imag(xy_complex)]

    def complex_to_point(xy_complex: complex, astype: type = int) -> tuple[Any, Any]:
        out = list(np.array(unpack(xy_complex)).astype(astype))
        out[1] = image.height - out[1]
        return tuple(out)

    def complexes_to_points(xy_complexes: Iterable[complex], astype: type = int) -> tuple[tuple[int, int], ...]:
        return tuple([complex_to_point(xy_complex, astype=astype) for xy_complex in xy_complexes])

    points = complexes_to_points(complexes, astype=float)
    tck, u = interpolate.splprep([
        list([float(x) for x, y in points]),
        list([float(y) for x, y in points])
    ], s=0)
    unew = np.arange(0, 1.001, 0.001)
    out = interpolate.splev(unew, tck)

    for xy in zip(*out):
            xy = tuple(np.array(xy).astype(int))
            draw.point(xy, (0, 0, 0, 255))
    return image

if __name__ == "__main__":
    Ring(128 * 16, 32 * 16, 1.5, .75, .2, [
        RingArm(n * 𝜏) for n in np.linspace(1 / 10, 1 + 1 / 10, 5 + 1, endpoint=False)
    ]).generate().show()
