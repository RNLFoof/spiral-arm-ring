import os
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from time import perf_counter
from typing import Iterable, Any, cast, Sequence, Optional, Self, Union

import numpy as np
import quads
import scipy
import zsil.internal
from PIL import Image
from PIL.ImageDraw import ImageDraw
# It would make more sense to use ASCII stuff here, but this is more fun
from numpy import pi as 𝜋
from scipy import interpolate
from tqdm import tqdm
from zsil import cool_stuff
from zsil.cool_stuff import GenerateFromNearestKeyParams

𝜏 = 2 * 𝜋
𝑖 = 1j


def unpack(xy_complex: complex) -> tuple[float, float]:
    return np.real(xy_complex), np.imag(xy_complex)


CanBecomePoint = Union[np.array, complex, tuple[float, float], quads.Point]


class PointAndMetadataThereof:

    def __init__(self, create_from: CanBecomePoint, arm: Optional["RingArm"] = None, ring: "Ring" = None,
                 bundle: Optional["Bundle"] = None):
        if isinstance(create_from, np.ndarray):
            self.as_array = create_from
        elif isinstance(create_from, complex):
            self.as_array = np.array(unpack(create_from))
        elif isinstance(create_from, tuple):
            self.as_array = np.array(create_from)
        elif isinstance(create_from, quads.Point):
            self.as_array = np.array((create_from.x, create_from.y))
        else:
            raise ValueError("You want me to make this out of WHAT")

        self.arm = arm
        self.ring = ring
        self.bundle = bundle

        if self.arm is not None and self.ring is None:
            self.ring = self.arm.ring
        if self.ring is None:
            raise Exception("You need to provide an arm or a ring! (preferably an arm)")

    def ill_make_a_point_outta_you(self, potential_point: Union[Self | CanBecomePoint]):
        if isinstance(potential_point, PointAndMetadataThereof):
            return potential_point
        return PointAndMetadataThereof(potential_point, ring=self.ring)

    def __add__(self, other: Union[Self | CanBecomePoint]) -> Self:
        return PointAndMetadataThereof(self.as_array + self.ill_make_a_point_outta_you(other).as_array,
                                       ring=self.ring)

    def __truediv__(self, other: float) -> Self:
        return PointAndMetadataThereof(self.as_array / other, ring=self.ring)

    def prepare_to_launch(self):
        return [
            self.as_tuple,
            self.distance_to_center,
            self.direction_to_center
        ]

    @cached_property
    def as_tuple(self) -> tuple[float, float]:
        return cast(
            tuple[float, float],
            tuple(self.as_array)
        )

    @cached_property
    def as_complex(self):
        return complex(*self.as_tuple)

    @cached_property
    def distance_to_center(self) -> float:
        return np.linalg.norm(self.ring.center.as_array - self.as_array)

    @cached_property
    def direction_to_center(self) -> float:
        return zsil.internal.point_direction(*self.as_tuple, *self.ring.center.as_tuple)


@dataclass
class PointBundle:
    center: PointAndMetadataThereof
    corner: PointAndMetadataThereof
    close_edge: PointAndMetadataThereof
    far_edge: PointAndMetadataThereof

    def __post_init__(self):
        for point in self.points():
            point.bundle = self

    def prepare_to_launch(self):
        for point in self.points():
            point.prepare_to_launch()
        return [
            self.current_arm_width
        ]

    def points(self):
        return [
            self.center,
            self.corner,
            self.close_edge,
            self.far_edge,
        ]

    @cached_property
    def current_arm_width(self) -> float:
        return arm_width_at(self.center.ring, self.center)


def arm_width_at(ring: "Ring", point: PointAndMetadataThereof) -> float:
    return np.interp(
        point.distance_to_center,
        [0, ring.width / 2],
        [ring.stroke_width, ring.stroke_width * 3]
    )


@dataclass
class RingArm:
    starting_radians: float
    ring: "Ring" = None

    def __post_init__(self):
        self._point_bundles: list[PointBundle] = []
        self._point_bundle_lookup: dict[tuple[float, float], PointBundle] = {}
        # self.ensure_point_bundles() would be here but is instead called by Ring after it plugs itself in

    def corner_points(self):
        return self._point_bundles

    def point_bundle_lookup(self):
        return self._point_bundle_lookup

    @cached_property
    def center_points(self) -> list[PointAndMetadataThereof]:
        largest_distance = self.ring.height * self.ring.outer_multiplier * 0.5
        shortest_distance = self.ring.height * self.ring.inner_multiplier * 0.5

        center_height = self.ring.height * 0.5j

        side_of_diagonal = angle_on_which_side_of_line(self.starting_radians, self.ring.side_dividing_line_radians)
        try:
            side_of_vertical = angle_on_which_side_of_line(self.starting_radians, 1 / 2 * 𝜋)
        except ParallelLineException:
            side_of_vertical = side_of_diagonal

        if side_of_diagonal == Side.LEFT:
            start = center_height
        else:
            start = self.ring.width + center_height

        if side_of_diagonal != side_of_vertical:
            start_from = .25 * 𝜏 if side_of_vertical == Side.LEFT else .75 * 𝜏
        else:
            start_from = self.starting_radians

        precision = 48
        spiral_complexes = []
        for radians in np.linspace(start_from, self.starting_radians + self.ring.rotations * 𝜏, num=precision):
            progress = scipy.interpolate.interp1d(
                [self.starting_radians, self.starting_radians + self.ring.rotations * 𝜏],
                [0, 1],
                fill_value="extrapolate"
            )(radians)
            distance = scipy.interpolate.interp1d(
                [0, 1],
                [largest_distance, shortest_distance],
                fill_value="extrapolate"
            )(progress)
            to_append = self.ring.center.as_complex + 𝑒𝑖(radians) * distance
            spiral_complexes.append(to_append)

        precision = 48
        buildup_gets_this_close = 0.01
        buildup_complexes = np.interp(
            np.linspace(0, buildup_gets_this_close, num=precision),
            [0, 1],
            [start, spiral_complexes[0]]
        )

        tip_complexeses = list(star_tips(self, self.ring, radians,
                                         self.ring.tip_closeness))  # ring.height * (ring.outer_multiplier - ring.inner_multiplier) / ring.rotations)
        out_tuples = []

        trim_around = spiral_complexes[-1]
        for spiral_index, spiral_complex in list(enumerate(spiral_complexes))[::-1]:
            if zsil.internal.point_distance(*unpack(trim_around),
                                            *unpack(spiral_complex)) > self.ring.spiral_trim:
                break
        spiral_complexes = spiral_complexes[:spiral_index]
        for tip_complexes_index, tip_complexes in enumerate(tip_complexeses):
            for tip_index, tip_complex in enumerate(tip_complexes):
                if zsil.internal.point_distance(*unpack(trim_around),
                                                *unpack(tip_complex)) > self.ring.starm_trim:
                    break
            tip_complexeses[tip_complexes_index] = tip_complexes[tip_index:]

        for tip_complexes in tip_complexeses:
            gonna_be_complexes = np.concatenate([buildup_complexes, spiral_complexes, tip_complexes], axis=0)
            seen = set()
            complexes = [[x, seen.add(np.round(x))][0] for x in gonna_be_complexes if np.round(x) not in seen]

            def complex_to_point(xy_complex: complex, astype: type = int) -> tuple[Any, Any]:
                out = list(np.array(unpack(xy_complex)).astype(astype))
                out[1] = self.ring.height - out[1]
                return tuple(out)

            def complexes_to_points(xy_complexes: Iterable[complex], astype: type = int) -> tuple[tuple[int, int], ...]:
                return tuple([complex_to_point(xy_complex, astype=astype) for xy_complex in xy_complexes])

            points = complexes_to_points(complexes, astype=float)
            tck, u = interpolate.splprep([
                list([float(x) for x, y in points]),
                list([float(y) for x, y in points])
            ], s=0)

            # The length of this proportionately affects how long the process takes
            # Which seems dumb. Like. it's all in a quadtree. shouldn't that be basically unaffected by size?
            # Maybe it like, skips layers if only use cell is used or something.
            unew = np.linspace(0, 1, self.ring.width)
            out_sideways = interpolate.splev(unew, tck)
            out_tuples += zip(*out_sideways)
            out = [PointAndMetadataThereof(point, arm=self) for point in out_tuples]

        return out

    def prepare_to_launch(self):
        for center_point in tqdm(self.center_points, "ensuring point bundles"):
            current_arm_width = arm_width_at(self.ring, center_point)
            corner_offset_distance = np.interp(
                center_point.distance_to_center,
                [0, self.ring.width / 2],
                [current_arm_width, 0]
            )
            corner_point = center_point + 𝑒𝑖(center_point.direction_to_center) * corner_offset_distance
            close_edge_point = center_point + 𝑒𝑖(center_point.direction_to_center) * current_arm_width
            far_edge_point = center_point + 𝑒𝑖(center_point.direction_to_center + 𝜋) * current_arm_width
            bundle = PointBundle(
                corner_point,
                corner_point,
                close_edge_point,
                far_edge_point
            )
            bundle.prepare_to_launch()
            self._point_bundles.append(bundle)
            for point in bundle.points():
                self._point_bundle_lookup[point.as_tuple] = bundle
        return


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
    ring_arms: Sequence[RingArm]
    spiral_trim: float = 0
    starm_trim: float = 0

    side_dividing_line_radians = 3 / 8 * 𝜏

    def __post_init__(self):
        for ring_arm in self.ring_arms:
            ring_arm.ring = self

    @cached_property
    def size(self) -> PointAndMetadataThereof:
        return PointAndMetadataThereof((self.width, self.height), ring=self)

    @cached_property
    def center(self) -> PointAndMetadataThereof:
        return self.size / 2

    def generate(self) -> Image:
        self.prepare_to_launch()

        image = Image.new("L", (self.width, self.height), 128)
        points = []

        for ring_arm in self.ring_arms:
            for coordinates in ring_arm.point_bundle_lookup():
                points.append(coordinates)

        # Rounding to int makes it about 2x faster, but it's so much cleaner without...
        # Rounding is separate to prevent rounding it OOB
        # usable_points = set(
        # (round(x, 1), round(y, 1)) for (x, y) in tqdm(points, "rounding usable points")
        #     (x, y) for (x, y) in tqdm(points, "rounding usable points")
        # )
        usable_points = set(
            (x, y) for (x, y) in tqdm(points, "trimming usable points") if
            0 <= x < image.width and 0 <= y < image.height
        )
        # TODO Move trimming stuff out of range into zsil?

        coordinates_to_go_over = set()
        rounded_usable_points = set((round(x), round(y)) for x, y in usable_points)
        for xy in tqdm(rounded_usable_points, "adding offsets"):
            # Doing just the edges because, since this is continuous, stuff adjacent should be gotten by points adjacent
            xy = PointAndMetadataThereof(xy, ring=self)
            offset_amount = int(np.ceil(arm_width_at(self, xy)))
            minimum_offset = -offset_amount
            maximum_offset = offset_amount
            offsets = list(range(minimum_offset, maximum_offset))
            for x_offset in offsets:
                for y_offset in offsets:
                    if len({minimum_offset, maximum_offset} - set(xy.as_tuple)) < 2:
                        continue
                    offset = np.array((x_offset, y_offset))
                    coordinates_to_go_over.add((xy + offset).as_tuple)
        coordinates_to_go_over = set(
            (round(x), round(y)) for (x, y) in tqdm(coordinates_to_go_over, "trimming coords to go over") if
            0 <= x < image.width and 0 <= y < image.height
        )
        coordinates_to_go_over = list(coordinates_to_go_over)

        all_point_bundle_lookup = {}
        for arm in self.ring_arms:
            all_point_bundle_lookup = {**all_point_bundle_lookup, **arm.point_bundle_lookup()}

        def key(p: cool_stuff.GenerateFromNearestKeyParams):
            me = PointAndMetadataThereof(p.coordinates, ring=self)
            closest = PointAndMetadataThereof(p.nearest_point, ring=self)
            if closest.as_tuple in all_point_bundle_lookup:
                bundle = all_point_bundle_lookup[closest.as_tuple]
            else:
                raise Exception("WHAT")

            sloping_towards_center = me.distance_to_center < bundle.corner.distance_to_center
            edge_point = bundle.close_edge if sloping_towards_center else bundle.far_edge
            distance_to_corner = np.linalg.norm(bundle.corner.as_array - me.as_array)
            distance_from_corner_to_edge = np.linalg.norm(edge_point.as_array - bundle.corner.as_array)

            color = round(np.interp(
                distance_to_corner,
                [0, distance_from_corner_to_edge],
                [0, 255]
            ))
            return color

        usable_points = list(usable_points)

        key(GenerateFromNearestKeyParams(image, (0, 0), list(all_point_bundle_lookup)[0]))

        cool_stuff.generate_from_nearest(image, usable_points, key,
                                         coordinates_to_go_over=coordinates_to_go_over
                                         )

        return image

    def prepare_to_launch(self):
        for arm in self.ring_arms:
            arm.prepare_to_launch()


def star_tips(arm: RingArm, ring: Ring, origin_radians: float, closest_allowed: float) -> tuple[list[complex], ...]:
    output = []
    center_height = ring.height * 0.5j
    center = ring.width * 0.5 + center_height
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
                            quads.Point(*unpack(new_complex)), count=1)
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
    # trim_summary()
    # exit()

    @contextmanager
    def catchtime() -> float:
        t1 = t2 = perf_counter()
        yield lambda: t2 - t1
        t2 = perf_counter()


    trims = []
    for spiral_trim in np.arange(0, 5, 0.25):
        for starm_trim in np.arange(0, 5, 0.25):
            trims.append((spiral_trim, starm_trim))
            break
    trims = sorted(trims, key=lambda x: x[0] + x[1])
    for multiplier in [4, 6, 8]:
        for spiral_trim, starm_trim in trims:
            filename = os.path.join(
                "trim_tests",
                f""""{str(float(spiral_trim)).replace('.', '_')}__{str(float(starm_trim)).replace('.', '_')}.png"""
            )
            print(filename)
            # if os.path.exists(filename):
            #     continue
            # continue
            with catchtime() as t:
                # multiplier = 4
                extra_vertical_room = 2
                ring = Ring(6 * 32 * multiplier,
                            extra_vertical_room * 32 * multiplier,
                            3 / 5,
                            0.9 / extra_vertical_room,
                            0.5 / extra_vertical_room,
                            3 / 4 * multiplier,
                            2 * multiplier,
                            [
                                RingArm(n * 𝜏 % 𝜏) for n in np.linspace(3 / 4, 1 + 3 / 4, 5, endpoint=False)
                            ],
                            spiral_trim * multiplier,
                            starm_trim * multiplier,
                            ).generate()
            print(f'Time: {t():.3f} seconds')
            ring.show()
            exit()
            ring.save(filename)
