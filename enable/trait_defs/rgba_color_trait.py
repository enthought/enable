# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Trait definition for an RGBA-based color, which is either:

* A tuple of the form (*red*,*green*,*blue*,*alpha*), where each component is
  in the range from 0.0 to 1.0
* An integer which in hexadecimal is of the form 0xAARRGGBB, where AA is alpha,
  RR is red, GG is green, and BB is blue.
"""

import numpy as np

from pyface.color import Color
from pyface.util.color_parser import ColorParseError, parse_text
from traits.api import TraitType
from traits.trait_base import SequenceTypes


class RGBAColor(TraitType):
    """ A Trait which casts Pyface Colors, strings and tuples to RGBA tuples.
    """

    def __init__(self, value="white", **metadata):
        default_value = self.validate(None, None, value)
        super().__init__(default_value, **metadata)

    def validate(self, object, name, value):
        if isinstance(value, Color):
            return value.rgba
        if isinstance(value, str):
            try:
                _, value = parse_text(value)
            except ColorParseError:
                self.error(object, name, value)
        is_array = isinstance(value, (np.ndarray, np.void))
        if is_array or isinstance(value, SequenceTypes):
            value = tuple(value)
            if len(value) == 3:
                value += (1.0,)
            if len(value) == 4:
                return value

        self.error(object, name, value)

    def info(self):
        return (
            "a Pyface Color, a #-hexadecimal rgb or rgba string,  a standard "
            "color name, or a sequence of RGBA or RGB values between 0 and 1"
        )

    def create_editor(self):
        from .ui.api import RGBAColorEditor
        return RGBAColorEditor()


# synonym for backwards compatibility
RGBAColorTrait = RGBAColor
