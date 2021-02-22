# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""
Defines markers classes, used by a variety of renderers.
"""

# Major library imports
from numpy import array, pi

# Enthought library imports
from traits.api import HasTraits, Bool, Instance, Trait
from traitsui.api import EnumEditor
from kiva.api import (
    CIRCLE_MARKER, CROSS_MARKER, DIAMOND_MARKER, DOT_MARKER, FILL_STROKE,
    INVERTED_TRIANGLE_MARKER, NO_MARKER, PIXEL_MARKER, PLUS_MARKER,
    SQUARE_MARKER, STROKE, TRIANGLE_MARKER
)

# Local imports
from .compiled_path import CompiledPath


class AbstractMarker(HasTraits):
    """ Abstract class for markers.
    """

    # How this marker is to be stroked (from kiva.api).
    # Since this needs to be a class variable, it can't be a trait.
    draw_mode = STROKE
    # draw_mode = Enum(FILL, EOF_FILL, STROKE, FILL_STROKE, EOF_FILL_STROKE)

    # The kiva marker type (from kiva.api).
    kiva_marker = NO_MARKER

    # Close the path object after drawing this marker?
    close_path = Bool(True)

    # Render the marker antialiased?  Some
    # markers render faster and look better if they are not anti-aliased..
    antialias = Bool(True)

    def add_to_path(self, path, size):
        """ Adds this marker's representation to *path*, scaled appropriately
        for *size*.

        Parameters
        ----------
        path : GraphicsContext
            The target for drawing the marker.
        size : number
            Size of the marker, in pixels
        """
        if self.close_path:
            self._add_to_path(path, size)
            path.close_path()
        else:
            self._add_to_path(path, size)

    def get_compiled_path(self, size):
        """ Returns a compiled path object that represents this marker, scaled
        appropriately for *size*.
        """
        raise NotImplementedError

    def _add_to_path(self, path, size):
        # subclasses must implement this method
        raise NotImplementedError


class SquareMarker(AbstractMarker):
    """ A marker that is a square.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = FILL_STROKE
    # The Kiva marker type. (Overrides AbstractMarker.)
    kiva_marker = SQUARE_MARKER
    # Do not render anti-aliased. (Overrides AbstractMarker.)
    antialias = False

    def _add_to_path(self, path, size):
        path.rect(-size, -size, size * 2, size * 2)


class DiamondMarker(AbstractMarker):
    """ A marker that is a diamond.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = FILL_STROKE
    # The Kiva marker type. (Overrides AbstractMarker.)
    kiva_marker = DIAMOND_MARKER
    # Do not render anti-aliased. (Overrides AbstractMarker.)
    antialias = False

    def _add_to_path(self, path, size):
        path.lines(array(((0, -size), (-size, 0), (0, size), (size, 0))))


class CircleMarker(AbstractMarker):
    """ A marker that is a circle.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = FILL_STROKE
    # The Kiva marker type. (Overrides AbstractMarker.)
    kiva_marker = CIRCLE_MARKER
    # Array of points in a circle
    circle_points = array(
        [
            [1.0, 0.0],
            [0.966, 0.259],
            [0.866, 0.5],
            [0.707, 0.707],
            [0.5, 0.866],
            [0.259, 0.966],
            [0.0, 1.0],
            [-0.259, 0.966],
            [-0.5, 0.866],
            [-0.707, 0.707],
            [-0.866, 0.5],
            [-0.966, 0.259],
            [-1.0, 0.0],
            [-0.966, -0.259],
            [-0.866, -0.5],
            [-0.707, -0.707],
            [-0.5, -0.866],
            [-0.259, -0.966],
            [0.0, -1.0],
            [0.259, -0.966],
            [0.5, -0.866],
            [0.707, -0.707],
            [0.866, -0.5],
            [0.966, -0.259],
            [1.0, 0.0],
        ]
    )

    def _add_to_path(self, path, size):
        if size <= 5:
            pts = self.circle_points[::3] * size
        elif size <= 10:
            pts = self.circle_points[::2] * size
        else:
            pts = self.circle_points * size
        path.lines(pts)


class TriangleMarker(AbstractMarker):
    """ A marker that is a triangle with one apex pointing up.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = FILL_STROKE
    # The Kiva marker type. (Overrides AbstractMarker.)
    kiva_marker = TRIANGLE_MARKER
    # Do not render anti-aliased. (Overrides AbstractMarker.)
    antialias = False

    def _add_to_path(self, path, size):
        path.lines(array(((-size, -size), (size, -size), (0, 0.732 * size))))


class Inverted_TriangleMarker(AbstractMarker):
    """ A marker that is a triangle with one apex pointing down.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = FILL_STROKE
    # The Kiva marker type. (Overrides AbstractMarker.)
    kiva_marker = INVERTED_TRIANGLE_MARKER
    # Do not render anti-aliased. (Overrides AbstractMarker.)
    antialias = False

    def _add_to_path(self, path, size):
        path.lines(array(((-size, size), (size, size), (0, -0.732 * size))))


class LeftTriangleMarker(AbstractMarker):
    """ A marker that is a triangle with one apex pointing left.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = FILL_STROKE
    # Do not render anti-aliased. (Overrides AbstractMarker.)
    antialias = True

    def _add_to_path(self, path, size):
        path.lines(array([(size, -size), (size, size), (-0.732 * size, 0)]))


class RightTriangleMarker(AbstractMarker):
    """ A marker that is a triangle with one apex pointing right.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = FILL_STROKE
    # Do not render anti-aliased. (Overrides AbstractMarker.)
    antialias = True

    def _add_to_path(self, path, size):
        path.lines(array([(-size, -size), (-size, size), (0.732 * size, 0)]))


class PentagonMarker(AbstractMarker):
    """ A marker that is a pentagon.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = FILL_STROKE
    # Do not render anti-aliased. (Overrides AbstractMarker.)
    antialias = True

    def _add_to_path(self, path, size):
        # xi = size * cos(2*pi*i/5. + pi/2), yi = size * sin(2*pi*i/5. + pi/2)
        path.lines(
            array(
                [
                    (0, size),
                    (0.951 * size, 0.309 * size),
                    (0.588 * size, -0.809 * size),
                    (-0.588 * size, -0.809 * size),
                    (-0.951 * size, 0.309 * size),
                ]
            )
        )


class Hexagon1Marker(AbstractMarker):
    """ A marker that is a hexagon, with the flat sides on the sides.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = FILL_STROKE
    # Do not render anti-aliased. (Overrides AbstractMarker.)
    antialias = True

    def _add_to_path(self, path, size):
        # xi = size * cos(2*pi*i/6.), yi = size * sin(2*pi*i/6.)
        path.lines(
            array(
                [
                    (size, 0),
                    (0.5 * size, 0.866 * size),
                    (-0.5 * size, 0.866 * size),
                    (-size, 0),
                    (-0.5 * size, -0.866 * size),
                    (0.5 * size, -0.866 * size),
                ]
            )
        )


class Hexagon2Marker(AbstractMarker):
    """ A marker that is a hexagon, with the flat sides on the top and bottom.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = FILL_STROKE
    # Do not render anti-aliased. (Overrides AbstractMarker.)
    antialias = True

    def _add_to_path(self, path, size):
        # Like Hexagon1Marker but with an offset of 30 deg.
        path.lines(
            array(
                [
                    (0.866 * size, 0.5 * size),
                    (0.0, size),
                    (-0.866 * size, 0.5 * size),
                    (-0.866 * size, -0.5 * size),
                    (0.0, -size),
                    (0.866 * size, -0.5 * size),
                ]
            )
        )


class PlusMarker(AbstractMarker):
    """ A marker that is a plus-sign.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = STROKE
    # The Kiva marker type. (Overrides AbstractMarker.)
    kiva_marker = PLUS_MARKER
    # Do not render anti-aliased. (Overrides AbstractMarker.)
    antialias = False

    def _add_to_path(self, path, size):
        path.move_to(0, -size)
        path.line_to(0, size)
        path.move_to(-size, 0)
        path.line_to(size, 0)


class CrossMarker(AbstractMarker):
    """ A marker that is an X.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = STROKE
    # The Kiva marker type. (Overrides AbstractMarker.)
    kiva_marker = CROSS_MARKER
    # Do not render anti-aliased. (Overrides AbstractMarker.)
    antialias = False

    def _add_to_path(self, path, size):
        path.move_to(-size, -size)
        path.line_to(size, size)
        path.move_to(size, -size)
        path.line_to(-size, size)


class StarMarker(AbstractMarker):
    """ A marker that is a (filled) star.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = FILL_STROKE
    # Do not render anti-aliased. (Overrides AbstractMarker.)
    antialias = True

    def _add_to_path(self, path, size):
        # Generated from
        # i = arange(10), thetai = 2*pi * i/10., ri = 0.75 + (-1)**i * 0.25
        # xi = ri * sin(thetai), yi = ri * cos(thetai)
        path.lines(
            array(
                [
                    (0.0, size),
                    (0.294 * size, 0.405 * size),
                    (0.951 * size, 0.309 * size),
                    (0.476 * size, -0.155 * size),
                    (0.588 * size, -0.809 * size),
                    (0, -0.5 * size),
                    (-0.588 * size, -0.809 * size),
                    (-0.476 * size, -0.155 * size),
                    (-0.951 * size, 0.309 * size),
                    (-0.294 * size, 0.405 * size),
                ]
            )
        )


class CrossPlusMarker(AbstractMarker):
    """ A marker that is an X and a + superimposed.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = STROKE
    # Do not render anti-aliased. (Overrides AbstractMarker.)
    antialias = True

    def _add_to_path(self, path, size):
        # Darw an X
        path.move_to(-size, -size)
        path.line_to(size, size)
        path.move_to(size, -size)
        path.line_to(-size, size)
        # Draw a +
        path.move_to(0, -size)
        path.line_to(0, size)
        path.move_to(-size, 0)
        path.line_to(size, 0)


class DotMarker(AbstractMarker):
    """ A marker that is a dot.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = FILL_STROKE
    # The Kiva marker type. (Overrides AbstractMarker.)
    kiva_marker = DOT_MARKER

    def _add_to_path(self, path, size):
        path.arc(0, 0, size, 0, 2 * pi)


class PixelMarker(AbstractMarker):
    """ A marker that is a pixel.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = STROKE
    # The Kiva marker type. (Overrides AbstractMarker.)
    kiva_marker = PIXEL_MARKER
    # Do not render anti-aliased. (Overrides AbstractMarker.)
    antialias = False

    def _add_to_path(self, path, size):
        # It's impossible to emulate a true Pixel Marker in a vector
        # system, so we just draw a sub-pixel square 1.0 unit across.
        path.rect(-0.5, -0.5, 1.0, 1.0)


class CustomMarker(AbstractMarker):
    """ A marker that is a custom shape.
    """

    # How this marker is to be stroked. (Overrides AbstractMarker.)
    draw_mode = STROKE
    # The Kiva marker type. (Overrides AbstractMarker.)
    kiva_marker = NO_MARKER

    # The custom path that represents this marker.
    path = Instance(CompiledPath)

    # Automatically scale **path** based on the input size parameter?
    # If False, then the path does not respond to the 'size' parameter!
    scale_path = Bool(True)

    def _add_to_path(self, path, size):
        if self.scale_path:
            path.save_ctm()
            path.scale_ctm(size, size)
        path.add_path(path)
        if self.scale_path:
            path.restore_ctm()

    def get_compiled_path(self, size):
        """ Returns a path instance.

        If **scale_path** is True, then the returned path is a new compiled
        path that is scaled based on *size*. If **scaled_path** is False,
        then this method just returns the current **path**.
        """
        if self.scale_path:
            newpath = CompiledPath()
            newpath.scale_ctm(size, size)
            newpath.add_path(self.path)
            return newpath
        else:
            return self.path


# String names for marker types.
marker_names = (
    "square",
    "circle",
    "triangle",
    "inverted_triangle",
    "left_triangle",
    "right_triangle",
    "pentagon",
    "hexagon",
    "hexagon2",
    "plus",
    "cross",
    "star",
    "cross_plus",
    "diamond",
    "dot",
    "pixel",
)

# Mapping of marker string names to classes.
MarkerNameDict = {
    "square": SquareMarker,
    "circle": CircleMarker,
    "triangle": TriangleMarker,
    "inverted_triangle": Inverted_TriangleMarker,
    "left_triangle": LeftTriangleMarker,
    "right_triangle": RightTriangleMarker,
    "pentagon": PentagonMarker,
    "hexagon": Hexagon1Marker,
    "hexagon2": Hexagon2Marker,
    "plus": PlusMarker,
    "cross": CrossMarker,
    "star": StarMarker,
    "cross_plus": CrossPlusMarker,
    "diamond": DiamondMarker,
    "dot": DotMarker,
    "pixel": PixelMarker,
    "custom": CustomMarker,
}

# A mapped trait that allows string naming of marker classes.
MarkerTrait = Trait(
    "square", MarkerNameDict, editor=EnumEditor(values=marker_names)
)

marker_trait = MarkerTrait
