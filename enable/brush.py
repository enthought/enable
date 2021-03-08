# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

""" Brush classes to make filling regions cleaner. """

from abc import abstractmethod
from operator import attrgetter

from numpy import array

from traits.api import (
    ABCHasStrictTraits, ArrayOrNone, Enum, Event, Float, HasStrictTraits,
    Instance, List, Range, Tuple, observe
)

from enable.colors import ColorTrait


class Brush(ABCHasStrictTraits):
    """ A brush describes how to fill the pixels of a region.

    To use a brush when drawing, call set_brush with the current gc as the
    argument.
    """

    #: A trait which fires when the brush is updated.
    updated = Event()

    @abstractmethod
    def set_brush(self, gc):
        """ Apply the brush settings to a graphics context.

        Parameters
        ----------
        gc : graphics context
            The graphics context to use the brush with.
        """
        raise NotImplementedError()

    @observe("+update")
    def observe_update_traits(self, event):
        self.updated = True


class ColorBrush(Brush):
    """ A simple brush that paints a solid color. """

    #: The color to brush the region with.
    color = ColorTrait("transparent", update=True)

    def set_brush(self, gc):
        """ Apply the brush settings to a graphics context.

        This sets the fill color of the GC to the specified color.

        Parameters
        ----------
        gc : graphics context
            The graphics context to use the brush with.
        """
        gc.set_fill_color(self.color_)


class ColorStop(HasStrictTraits):
    """ A point on a gradient with a fixed color. """

    #: The position of the color stop in the gradient, between 0.0 and 1.0.
    offset = Range(0.0, 1.0, update=True)

    #: The color at the color stop.
    color = ColorTrait("transparent", update=True)

    #: A trait which fires when the color stop is updated.
    updated = Event()

    def to_array(self):
        """ Return an array which represents the color stop.

        This is the raw form of the color stop required by Kiva.

        Returns
        -------
        stop_array : numpy array
            Return an array of (offset, r, b, b, a).
        """
        return array([self.offset] + list(self.color_))

    @observe("+update")
    def observe_update_traits(self, event):
        self.updated = True


class Gradient(HasStrictTraits):
    """ A color gradient. """

    #: The sequence of color stops for the gradient.
    stops = List(Instance(ColorStop))

    #: A trait which fires when the gradient is updated.
    updated = Event()

    #: A temporary cache for the stop array.
    _array_cache = ArrayOrNone()

    def to_array(self):
        """ Return a sorted list of stop arrays.

        This is the raw form of the stops required by Kiva.

        Returns
        -------
        stop_array_list : arrays
            A list of array of (offset, r, b, b, a) values corresponding to
            the color stops.  This array should not be mutated.
        """
        if self._array_cache is None:
            self._array_cache = array([
                stop.to_array()
                for stop in sorted(self.stops, key=attrgetter("offset"))
            ])
        return self._array_cache

    @observe("stops.items.updated")
    def observe_stops(self, event):
        self._array_cache = None
        self.updated = True

    def _stops_default(self):
        return [
            ColorStop(offset=0.0, color="white"),
            ColorStop(offset=1.0, color="black"),
        ]


class GradientBrush(Brush):
    """ A brush that paints a color gradient. """

    #: The sequence of color stops for the gradient.
    gradient = Instance(Gradient, args=(), allow_none=False, update=True)

    #: How the gradient extends beyond the main area of the gradient.
    #:
    #: pad
    #:    Extend using the first and last colors.
    #: reflect
    #:    Extend using a reflection of the gradient.
    #: repeat
    #:    Extend by repeating the gradient.
    spread_method = Enum("pad", "reflect", "repeat", update=True)

    #: The coordinate space used by the gradient points.
    #:
    #: userSpaceOnUse
    #:     Coordinates are specified in the current graph context's coordinate
    #:     system.
    #: objectBoundingBox
    #:     Coordinates are specified between 0.0 and 1.0 and are relative to
    #:     the bounding box of the path being filled.
    units = Enum("userSpaceOnUse", "objectBoundingBox", update=True)

    @observe("gradient:updated")
    def observe_stops(self, event):
        self.updated = True


class LinearGradientBrush(GradientBrush):
    """ A brush that paints a linear color gradient. """

    #: The start point of the linear gradient.
    start = Tuple(Float, Float, update=True)

    #: The stop point of the linear gradient.
    end = Tuple(Float, Float, update=True)

    def set_brush(self, gc):
        """ Apply the brush settings to a graphics context.

        This calls linear_gradient on the GC with the appropriate values.

        Parameters
        ----------
        gc : graphics context
            The graphics context to use the brush with.
        """
        gc.linear_gradient(
            *self.start,
            *self.end,
            self.gradient.to_array(),
            self.spread_method,
            self.units,
        )


class RadialGradientBrush(GradientBrush):
    """ A brush that paints a radial color gradient. """

    #: The center point of the radial gradient.
    center = Tuple(Float, Float, update=True)

    #: The radius of the radial gradient.
    radius = Float(update=True)

    #: The focus point of the radial gradient.
    focus = Tuple(Float, Float, update=True)

    def set_brush(self, gc):
        """ Apply the brush settings to a graphics context.

        This calls radial_gradient on the GC with the appropriate values.

        Parameters
        ----------
        gc : graphics context
            The graphics context to use the brush with.
        """
        gc.radial_gradient(
            *self.center,
            self.radius,
            *self.focus,
            self.gradient.to_array(),
            self.spread_method,
            self.units,
        )
