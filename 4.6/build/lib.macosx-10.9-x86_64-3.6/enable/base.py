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
Define a base set of constants and functions used by the remainder of the
Enable package.
"""
# -----------------------------------------------------------------------------
#  Functions defined: bounding_box
#                     intersect_coordinates
#                     union_coordinates
#                     intersect_bounds
#                     union_bounds
#                     disjoint_intersect_coordinates
#                     does_disjoint_intersect_coordinates
#                     bounding_coordinates
#                     bounds_to_coordinates
#                     coordinates_to_bounds
#                     coordinates_to_size
#                     add_rectangles
#                     xy_in_bounds
#                     gc_image_for
#                     send_event_to
#                     subclasses_of
# -----------------------------------------------------------------------------

# Enthought library imports
from traits.api import TraitError

from kiva.api import (
    BOLD, DECORATIVE, DEFAULT, ITALIC, MODERN, NORMAL, ROMAN, SCRIPT, SWISS,
    Font,
)

from .colors import color_table, transparent_color

# Special 'empty rectangle' indicator:
empty_rectangle = -1

# Used to offset positions by half a pixel and bounding width/height by 1.
# TODO: Resolve this in a more intelligent manner.
half_pixel_bounds_inset = (0.5, 0.5, -1.0, -1.0)

# Positions:
TOP = 32
VCENTER = 16
BOTTOM = 8
LEFT = 4
HCENTER = 2
RIGHT = 1

TOP_LEFT = TOP + LEFT
TOP_RIGHT = TOP + RIGHT
BOTTOM_LEFT = BOTTOM + LEFT
BOTTOM_RIGHT = BOTTOM + RIGHT

# -----------------------------------------------------------------------------
# Helper font functions
# -----------------------------------------------------------------------------

font_families = {
    "default": DEFAULT,
    "decorative": DECORATIVE,
    "roman": ROMAN,
    "script": SCRIPT,
    "swiss": SWISS,
    "modern": MODERN,
}
font_styles = {"italic": ITALIC}
font_weights = {"bold": BOLD}
font_noise = ["pt", "point", "family"]


def str_to_font(object, name, value):
    "Converts a (somewhat) free-form string into a valid Font object."
    # FIXME: Make this less free-form and more well-defined.
    try:
        point_size = 10
        family = SWISS
        style = NORMAL
        weight = NORMAL
        underline = 0
        face_name = []
        for word in value.split():
            lword = word.lower()
            if lword in font_families:
                family = font_families[lword]
            elif lword in font_styles:
                style = font_styles[lword]
            elif lword in font_weights:
                weight = font_weights[lword]
            elif lword == "underline":
                underline = 1
            elif lword not in font_noise:
                try:
                    point_size = int(lword)
                except Exception:
                    face_name.append(word)
        return Font(
            face_name=" ".join(face_name),
            size=point_size,
            family=family,
            weight=weight,
            style=style,
            underline=underline,
        )
    except Exception:
        pass
    raise TraitError(object, name, "a font descriptor string", repr(value))


str_to_font.info = (
    "a string describing a font (e.g. '12 pt bold italic "
    + "swiss family Arial' or 'default 12')"
)

# Pick a default font that should work on all platforms.
default_font_name = "modern 10"
default_font = str_to_font(None, None, default_font_name)


def bounding_box(components):
    "Compute the bounding box for a set of components"
    bxl, byb, bxr, byt = bounds_to_coordinates(components[0].bounds)
    for component in components[1:]:
        xl, yb, xr, yt = bounds_to_coordinates(component.bounds)
        bxl = min(bxl, xl)
        byb = min(byb, yb)
        bxr = max(bxr, xr)
        byt = max(byt, yt)
    return (bxl, byb, bxr, byt)


def intersect_coordinates(coordinates1, coordinates2):
    "Compute the intersection of two coordinate based rectangles"
    if (coordinates1 is empty_rectangle) or (coordinates2 is empty_rectangle):
        return empty_rectangle
    xl1, yb1, xr1, yt1 = coordinates1
    xl2, yb2, xr2, yt2 = coordinates2
    xl = max(xl1, xl2)
    yb = max(yb1, yb2)
    xr = min(xr1, xr2)
    yt = min(yt1, yt2)
    if (xr > xl) and (yt > yb):
        return (xl, yb, xr, yt)
    return empty_rectangle


def intersect_bounds(bounds1, bounds2):
    "Compute the intersection of two bounds rectangles"
    if (bounds1 is empty_rectangle) or (bounds2 is empty_rectangle):
        return empty_rectangle

    intersection = intersect_coordinates(
        bounds_to_coordinates(bounds1), bounds_to_coordinates(bounds2)
    )
    if intersection is empty_rectangle:
        return empty_rectangle
    xl, yb, xr, yt = intersection
    return (xl, yb, xr - xl, yt - yb)


def union_coordinates(coordinates1, coordinates2):
    "Compute the union of two coordinate based rectangles"
    if coordinates1 is empty_rectangle:
        return coordinates2
    elif coordinates2 is empty_rectangle:
        return coordinates1
    xl1, yb1, xr1, yt1 = coordinates1
    xl2, yb2, xr2, yt2 = coordinates2
    return (min(xl1, xl2), min(yb1, yb2), max(xr1, xr2), max(yt1, yt2))


def union_bounds(bounds1, bounds2):
    "Compute the union of two bounds rectangles"
    xl, yb, xr, yt = union_coordinates(
        bounds_to_coordinates(bounds1), bounds_to_coordinates(bounds2)
    )
    if xl is None:
        return empty_rectangle
    return (xl, yb, xr - xl, yt - yb)


def does_disjoint_intersect_coordinates(coordinates_list, coordinates):
    """ Return whether a rectangle intersects a disjoint set of rectangles
    anywhere
    """
    # If new rectangle is empty, the result is empty:
    if coordinates is empty_rectangle:
        return False

    # If we have an 'infinite' area, then return the new rectangle:
    if coordinates_list is None:
        return True

    # Intersect the new rectangle against each rectangle in the list until an
    # non_empty intersection is found:
    xl1, yb1, xr1, yt1 = coordinates
    for xl2, yb2, xr2, yt2 in coordinates_list:
        if (min(xr1, xr2) > max(xl1, xl2)) and (min(yt1, yt2) > max(yb1, yb2)):
            return True
    return False


def bounding_coordinates(coordinates_list):
    "Return the bounding rectangle for a list of rectangles"
    if coordinates_list is None:
        return None
    if len(coordinates_list) == 0:
        return empty_rectangle
    xl, yb, xr, yt = 1.0e10, 1.0e10, -1.0e10, -1.0e10
    for xl1, yb1, xr1, yt1 in coordinates_list:
        xl = min(xl, xl1)
        yb = min(yb, yb1)
        xr = max(xr, xr1)
        yt = max(yt, yt1)
    return (xl, yb, xr, yt)


def bounds_to_coordinates(bounds):
    "Convert a bounds rectangle to a coordinate rectangle"
    x, y, dx, dy = bounds
    return (x, y, x + dx, y + dy)


def coordinates_to_bounds(coordinates):
    "Convert a coordinates rectangle to a bounds rectangle"
    xl, yb, xr, yt = coordinates
    return (xl, yb, xr - xl, yt - yb)


def coordinates_to_size(coordinates):
    "Convert a coordinates rectangle to a size tuple"
    xl, yb, xr, yt = coordinates
    return (xr - xl, yt - yb)


def add_rectangles(rectangle1, rectangle2):
    "Add two bounds or coordinate rectangles"
    return (
        rectangle1[0] + rectangle2[0],
        rectangle1[1] + rectangle2[1],
        rectangle1[2] + rectangle2[2],
        rectangle1[3] + rectangle2[3],
    )


def xy_in_bounds(x, y, bounds):
    "Test whether a specified (x,y) point is in a specified bounds"
    x0, y0, dx, dy = bounds
    return (x0 <= x < x0 + dx) and (y0 <= y < y0 + dy)


def send_event_to(components, event_name, event):
    "Send an event to a specified set of components until it is 'handled'"
    pre_event_name = "pre_" + event_name
    for component in components:
        setattr(component, pre_event_name, event)
        if event.handled:
            return len(components)
    for i in range(len(components) - 1, -1, -1):
        setattr(components[i], event_name, event)
        if event.handled:
            return i
    return 0


def subclasses_of(klass):
    "Generate all of the classes (and subclasses) for a specified class"
    yield klass
    for subclass in klass.__bases__:
        for result in subclasses_of(subclass):
            yield result


class IDroppedOnHandler:
    "Interface for draggable objects that handle the 'dropped_on' event"

    def was_dropped_on(self, component, event):
        raise NotImplementedError
