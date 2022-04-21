# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Trait definition for an RGBA-based color

An RGBA-based color is a tuple of the form (*red*, *green*, *blue*, *alpha*),
where each component is in the range from 0.0 to 1.0, which is the color
representation used by Kiva for drawing.

Two traits are made available, a casting trait which takes many different color
representations and converts them to a tuple, and a mapped version of the trait
which holds the unconverted value and has a shadow trait holding the RGBA tuple
value.
"""
import copy

import numpy as np

from pyface.color import Color as PyfaceColor
from pyface.util.color_helpers import ints_to_channels
from pyface.util.color_parser import color_table, parse_text
from traits.api import DefaultValue, TraitType
from traits.trait_base import SequenceTypes

# Placeholders for system- and toolkit-specific UI colors; the
# toolkit-dependent code below will fill these with the appropriate
# values.  These hardcoded defaults are for the Windows Classic
# theme.
color_table["syswindow"] = (0.83137, 0.81569, 0.78431, 1.0)


def convert_to_color_tuple(value):
    """Convert various color representations to Kiva color tuple.

    This can accept integers of the form 0xRRGGBB, strings which
    can be parsed by Pyface's color parser, (r, g, b) and (r, g, b, a) tuples
    with either float 0.0 to 1.0 or int 0 to 255, Pyface Color objects, and
    toolkit color objects.
    """
    if isinstance(value, int):
        if 0 <= value <= 0xFFFFFF:
            return (
                (value >> 16) / 255.0,
                ((value >> 8) & 0xFF) / 255.0,
                (value & 0xFF) / 255.0,
                1.0,
            )
        else:
            raise ValueError(
                f"Integer value must be of the form 0xRRGGBB"
            )

    if isinstance(value, str):
        _, value = parse_text(value)

    is_array = isinstance(value, (np.ndarray, np.void))
    if is_array or isinstance(value, SequenceTypes):
        value = tuple(value)
        if len(value) not in {3, 4}:
            raise ValueError("Sequence must have length 3 or 4.")
        if all(isinstance(x, (int, np.integer)) for x in value):
            if len(value) == 3:
                value += (255,)
            if all(0 <= x < 256 for x in value):
                value = ints_to_channels(value)
            else:
                raise ValueError(
                    f"Integer sequence values not in range 0 to 255: {value}"
                )
        if len(value) == 3:
            value += (1.0,)
        if all(0 <= x <= 1.0 for x in value):
            return value
        else:
            raise ValueError(
                f"Float sequence values not in range 0 to 1: {value}"
            )

    if not isinstance(value, PyfaceColor):
        # assume that it is a toolkit color object
        value = PyfaceColor.from_toolkit(value)
    return value.rgba


class RGBAColor(TraitType):
    """ A Trait which casts Pyface Colors, strings and tuples to RGBA tuples.
    """

    #: Default values should be a tuple of floats.
    default_value_type = DefaultValue.constant

    def __init__(self, value="white", **metadata):
        default_value = convert_to_color_tuple(value)
        super().__init__(default_value, **metadata)

    def validate(self, object, name, value):
        try:
            return convert_to_color_tuple(value)
        except Exception:
            self.error(object, name, value)

        self.error(object, name, value)

    def info(self):
        return (
            "a Pyface Color, a #-hexadecimal rgb or rgba string, a CSS "
            "color representation, a sequence of RGBA or RGB floats "
            "between 0.0 and 1.0 or ints between 0 and 255, or an integer of "
            "the form 0xRRGGBB"
        )

    def create_editor(self):
        from .ui.api import RGBAColorEditor
        return RGBAColorEditor()


class MappedRGBAColor(RGBAColor):
    """A mapped trait that maps various color representations to RGBA tuples.
    """

    is_mapped = True

    def validate(self, object, name, value):
        try:
            convert_to_color_tuple(value)
        except Exception:
            self.error(object, name, value)

        return value

    def mapped_value(self, value):
        """Return the mapped RGBA value."""
        return convert_to_color_tuple(value)

    def post_setattr(self, object, name, value):
        """Set the shadow trait after setting the current value."""
        setattr(object, name + '_', self.mapped_value(value))


# synonyms for backwards compatibility
RGBAColorTrait = RGBAColor
ColorTrait = MappedRGBAColor("black")

black_color_trait = MappedRGBAColor("black")
white_color_trait = MappedRGBAColor("white")
transparent_color_trait = MappedRGBAColor("transparent")
