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
Define the base Enable object traits
"""

# Major library imports
from numpy import array, ndarray

# Enthought library imports
from kiva.trait_defs.api import KivaFont
from traits.api import List, PrefixList, PrefixMap, Range, Trait, TraitFactory
from traitsui.api import ImageEnumEditor, EnumEditor

# Try to get the CList trait; for traits 2 backwards compatibility, fall back
# to a normal List trait if we can't import it
try:
    from traits.api import CList
except ImportError:
    CList = List

# Relative imports
import enable.base as base
from .base import default_font_name

# -----------------------------------------------------------------------------
#  Constants:
# -----------------------------------------------------------------------------

# numpy 'array' type:
ArrayType = ndarray

# Basic sequence types:
basic_sequence_types = (list, tuple)

# Sequence types:
sequence_types = [ArrayType, list, tuple]

# Valid pointer shape names:
pointer_shapes = [
    "arrow",
    "right arrow",
    "blank",
    "bullseye",
    "char",
    "cross",
    "hand",
    "ibeam",
    "left button",
    "magnifier",
    "middle button",
    "no entry",
    "paint brush",
    "pencil",
    "point left",
    "point right",
    "question arrow",
    "right button",
    "size top",
    "size bottom",
    "size left",
    "size right",
    "size top right",
    "size bottom left",
    "size top left",
    "size bottom right",
    "sizing",
    "spray can",
    "wait",
    "watch",
    "arrow wait",
]

# Cursor styles:
CURSOR_X = 1
CURSOR_Y = 2

cursor_styles = {
    "default": -1,
    "none": 0,
    "horizontal": CURSOR_Y,
    "vertical": CURSOR_X,
    "both": CURSOR_X | CURSOR_Y,
}

border_size_editor = ImageEnumEditor(
    values=[x for x in range(9)], suffix="_weight", cols=3, module=base
)


# -----------------------------------------------------------------------------
# LineStyle trait
# -----------------------------------------------------------------------------

# Privates used for specification of line style trait.
__line_style_trait_values = {
    "solid": None,
    "dot dash": array([3.0, 5.0, 9.0, 5.0]),
    "dash": array([6.0, 6.0]),
    "dot": array([2.0, 2.0]),
    "long dash": array([9.0, 5.0]),
}
__line_style_trait_map_keys = list(__line_style_trait_values.keys())
LineStyleEditor = EnumEditor(values=__line_style_trait_map_keys)


def __line_style_trait(value="solid", **metadata):
    return Trait(
        value, __line_style_trait_values, editor=LineStyleEditor, **metadata
    )


# A mapped trait for use in specification of line style attributes.
LineStyle = TraitFactory(__line_style_trait)


# -----------------------------------------------------------------------------
#  Trait definitions:
# -----------------------------------------------------------------------------

# Font trait:
font_trait = KivaFont(default_font_name)

# Bounds trait
bounds_trait = CList([0.0, 0.0])  # (w,h)
coordinate_trait = CList([0.0, 0.0])  # (x,y)

# bounds_trait = Trait((0.0, 0.0, 20.0, 20.0), valid_bounds,
#                      editor=bounds_editor)

# Component minimum size trait
# PZW: Make these just floats, or maybe remove them altogether.
ComponentMinSize = Range(0.0, 99999.0)
ComponentMaxSize = ComponentMinSize(99999.0)

# Pointer shape trait:
Pointer = PrefixList(pointer_shapes, default_value="arrow")

# Cursor style trait:
cursor_style_trait = PrefixMap(cursor_styles, default_value="default")

spacing_trait = Range(0, 63, value=4)
padding_trait = Range(0, 63, value=4)
margin_trait = Range(0, 63)
border_size_trait = Range(0, 8, editor=border_size_editor)

# Time interval trait:
TimeInterval = Trait(None, None, Range(0.0, 3600.0))

# Stretch traits:
Stretch = Range(0.0, 1.0, value=1.0)
NoStretch = Stretch(0.0)
