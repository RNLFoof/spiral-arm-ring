import os
import re
from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Iterable, Any

import numpy as np
import quads
import scipy
import zsil.internal
from PIL import Image
from PIL.ImageDraw import ImageDraw
# It would make more sense to use ASCII stuff here, but this is more fun
from numpy import pi as 𝜋
from scipy import interpolate
from zsil import cool_stuff

𝜏 = 2 * 𝜋
𝑖 = 1j


def unpack(xy_complex: complex) -> list[float]:
    return [np.real(xy_complex), np.imag(xy_complex)]


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
    stroke_width: float
    tip_closeness: float
    ring_arms: Iterable[RingArm]
    spiral_trim: float = 0
    starm_trim: float = 0

    side_dividing_line_radians = 3 / 8 * 𝜏

    def generate(self) -> Image:
        image = Image.new("RGBA", (self.width, self.height))
        points = []

        for ring_arm in self.ring_arms:
            points += add_an_arm(image, ring_arm, self)

        usable_points = set(
            (round(x), round(y)) for (x, y) in points if 0 <= x < image.width and 0 <= y < image.height
        )

        max_distance = 10

        def key(p: cool_stuff.GenerateFromNearestKeyParams):
            # if p.distance() > self.stroke_width:
            #     return None
            color = round(
                min(
                    self.stroke_width,
                    p.distance()
                ) / self.stroke_width * 255
            )
            return (
                color, color, color, 255
            )

        cool_stuff.generate_from_nearest(image, usable_points, key)

        return image


def add_an_arm(image: Image, arm: RingArm, ring: Ring):
    largest_distance = ring.height * ring.outer_multiplier * 0.5
    shortest_distance = ring.height * ring.inner_multiplier * 0.5
    draw = ImageDraw(image)

    center_height = ring.height * 0.5j
    center = ring.width * 0.5 + center_height

    side_of_diagonal = angle_on_which_side_of_line(arm.starting_radians, ring.side_dividing_line_radians)
    try:
        side_of_vertical = angle_on_which_side_of_line(arm.starting_radians, 1 / 2 * 𝜋)
    except ParallelLineException:
        side_of_vertical = side_of_diagonal

    if side_of_diagonal == Side.LEFT:
        start = center_height
    else:
        start = ring.width + center_height

    if side_of_diagonal != side_of_vertical:
        start_from = .25 * 𝜏 if side_of_vertical == Side.LEFT else .75 * 𝜏
    else:
        start_from = arm.starting_radians

    precision = 48
    spiral_complexes = []
    for radians in np.linspace(start_from, arm.starting_radians + ring.rotations * 𝜏, num=precision):
        progress = scipy.interpolate.interp1d(
            [arm.starting_radians, arm.starting_radians + ring.rotations * 𝜏],
            [0, 1],
            fill_value="extrapolate"
        )(radians)
        distance = scipy.interpolate.interp1d(
            [0, 1],
            [largest_distance, shortest_distance],
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

    tip_complexeses = list(star_tips(arm, ring, radians,
                                     ring.tip_closeness))  # ring.height * (ring.outer_multiplier - ring.inner_multiplier) / ring.rotations)
    out = []

    trim_around = spiral_complexes[-1]
    for spiral_index, spiral_complex in list(enumerate(spiral_complexes))[::-1]:
        if zsil.internal.point_distance(*unpack(trim_around),
                                        *unpack(spiral_complex)) > ring.spiral_trim:
            break
    spiral_complexes = spiral_complexes[:spiral_index]
    for tip_complexes_index, tip_complexes in enumerate(tip_complexeses):
        for tip_index, tip_complex in enumerate(tip_complexes):
            if zsil.internal.point_distance(*unpack(trim_around),
                                            *unpack(tip_complex)) > ring.starm_trim:
                break
        tip_complexeses[tip_complexes_index] = tip_complexes[tip_index:]

    for tip_complexes in tip_complexeses:
        gonna_be_complexes = np.concatenate([buildup_complexes, spiral_complexes, tip_complexes], axis=0)
        seen = set()
        complexes = [[x, seen.add(np.round(x))][0] for x in gonna_be_complexes if np.round(x) not in seen]

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
        unew = np.linspace(0, 1, image.width)
        out_sideways = interpolate.splev(unew, tck)
        out += zip(*out_sideways)

    return out


def star_tips(arm: RingArm, ring: Ring, origin_radians: float, closest_allowed: float) -> tuple[list[complex], ...]:
    output = []
    center_height = ring.height * 0.5j
    center = ring.width * 0.5 + center_height
    largest_distance = ring.height * ring.outer_multiplier * 0.5
    shortest_distance = ring.height * ring.inner_multiplier * 0.5

    class Tip:
        ordered_complexes: list[complex] = None
        complex_lookup: quads.QuadTree = None

        def __init__(self, origin_radians: float):
            self.origin_radians = origin_radians
            self.ordered_complexes = []
            self.complex_lookup = quads.QuadTree(unpack(center), ring.width, ring.height)

        def append(self, new_complex: complex) -> None:
            self.ordered_complexes.append(new_complex)
            self.complex_lookup.insert(unpack(new_complex), new_complex)

    tips: list[Tip] = []
    for reverse in [-1, 1]:
        currently_expanding_tips: list[Tip] = []
        for index in range(5):
            tip = Tip(origin_radians + 𝜏 / 5 * index)
            currently_expanding_tips.append(tip)
        output.append(currently_expanding_tips[0].ordered_complexes)
        tips += currently_expanding_tips

        get_me_outta_this_thing = False
        for progress in np.linspace(0, 1, 32):
            if get_me_outta_this_thing: break

            for expanding_tip_index, expanding_tip in enumerate(currently_expanding_tips):
                if get_me_outta_this_thing: break

                origin = center + 𝑒𝑖(expanding_tip.origin_radians) * shortest_distance
                destination = center + 𝑒𝑖(expanding_tip.origin_radians + 𝜏 * 2 / 5 * reverse) * shortest_distance
                new_complex: complex = complex(scipy.interpolate.interp1d(
                    [0, 1],
                    [origin, destination],
                )(progress))

                if expanding_tip_index == 0:  # Only bother doing this once
                    for testing_tip in tips:
                        if testing_tip.origin_radians == expanding_tip.origin_radians:
                            continue
                        closest_maybe: list[quads.Point] = testing_tip.complex_lookup.nearest_neighbors(
                            unpack(new_complex), count=1)
                        if len(closest_maybe) != 0 and zsil.internal.point_distance(*unpack(new_complex),
                                                                                    closest_maybe[0].x, closest_maybe[
                                                                                        0].y) < closest_allowed:
                            get_me_outta_this_thing = True
                            break
                expanding_tip.append(new_complex)

    return tuple(output)


def trim_summary():
    trim_test_directory = "trim_tests"
    xs = set()
    ys = set()

    def parts_from_name(filename: str) -> tuple[str, str]:
        output = tuple(re.findall("\d+_\d+", filename))
        assert len(output) == 2
        return output

    individual_image_max_size = (0, 0)
    for filename in os.listdir(trim_test_directory):
        x, y = parts_from_name(filename)
        xs.add(x)
        ys.add(y)

        image = Image.open(os.path.join(trim_test_directory, filename))
        if image.size[0] > individual_image_max_size[0]:
            individual_image_max_size = image.size

    xs = sorted(list(xs))
    ys = sorted(list(ys))

    output = Image.new("RGB",
                       (
                           individual_image_max_size[0] * len(xs),
                           individual_image_max_size[1] * len(ys)
                       ),
                       "white"
                       )
    for filename in os.listdir(trim_test_directory):
        xy = parts_from_name(filename)
        x, y = xy

        x_position = xs.index(x)
        y_position = ys.index(y)

        floaty_xy = [float(axis.replace("_", ".")) for axis in xy]
        spiral_trim, starm_trim = floaty_xy

        pastee = Image.open(os.path.join(trim_test_directory, filename)).convert("RGBA")
        pastee = zsil.cool_stuff.enlargeable_thumbnail(pastee, individual_image_max_size, Image.BOX)
        draw = ImageDraw(pastee)
        draw.text((3, 0), f"{spiral_trim=}, {starm_trim=}", "black")

        rectangle_image = Image.new("RGBA", pastee.size, 0)
        rectangle_draw = ImageDraw(rectangle_image)
        rectangle_draw.rectangle((0, 0, *pastee.size), (0, 0, 0, 0), "black", 1)

        pastee.alpha_composite(rectangle_image)

        output.paste(
            pastee,
            (
                individual_image_max_size[0] * x_position,
                individual_image_max_size[1] * y_position
            )
        )

    output.save("all.png")


if __name__ == "__main__":
    trim_summary()
    exit()


    @contextmanager
    def catchtime() -> float:
        t1 = t2 = perf_counter()
        yield lambda: t2 - t1
        t2 = perf_counter()


    trims = []
    for spiral_trim in np.arange(0, 5, 0.25):
        for starm_trim in np.arange(0, 5, 0.25):
            trims.append((spiral_trim, starm_trim))
    trims = sorted(trims, key=lambda x: x[0] + x[1])
    for multiplier in [6, 8]:
        for spiral_trim, starm_trim in trims:
            filename = os.path.join(
                "trim_tests",
                f"{str(float(spiral_trim)).replace('.', '_')}__{str(float(starm_trim)).replace('.', '_')}.png"
            )
            print(filename)
            # if os.path.exists(filename):
            #     continue
            # continue
            with catchtime() as t:
                # multiplier = 4
                ring = Ring(6 * 32 * multiplier,
                            32 * multiplier,
                            3 / 5,
                            0.9,
                            0.6,
                            3 / 4 * multiplier,
                            2 * multiplier,
                            [
                                RingArm(n * 𝜏 % 𝜏) for n in np.linspace(3 / 4, 1 + 3 / 4, 5, endpoint=False)
                            ],
                            spiral_trim * multiplier,
                            starm_trim * multiplier,
                            ).generate()
            print(f'Time: {t():.3f} seconds')
            ring.save(filename)
