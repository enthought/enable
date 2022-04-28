# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Trait definition for a wxPython-based Kiva font.
"""

from pyface.font import Font as PyfaceFont
from traits.api import DefaultValue, TraitError, TraitType, NoDefaultSpecified

import kiva.constants as kc
from kiva.fonttools.font import Font, FontParseError, simple_parser


#: Expected attributes on the Font class.
font_attrs = [
    'face_name', 'size', 'family', 'weight', 'style', 'underline', 'encoding',
]

#: Mapping from Pyface Font generic family names to corresponding constants.
pyface_family_to_kiva_family = {
    'default': kc.DEFAULT,
    'fantasy': kc.DECORATIVE,
    'decorative': kc.DECORATIVE,
    'serif': kc.ROMAN,
    'roman': kc.ROMAN,
    'cursive': kc.SCRIPT,
    'script': kc.SCRIPT,
    'sans-serif': kc.SWISS,
    'swiss': kc.SWISS,
    'monospace': kc.MODERN,
    'modern': kc.MODERN,
    'typewriter': kc.TELETYPE,
    'teletype': kc.TELETYPE,
}


def pyface_font_to_font(font):
    """Convert a Pyface font to an equivalent Kiva Font.

    This ignores stretch and some options like small caps and strikethrough
    as the Kiva font object can't represent these at the moment.

    Parameters
    ----------
    font : Pyface Font instance
        The font to convert.

    Returns
    -------
    font : Kiva Font instance
        The resulting Kiva Font object.
    """
    face_name = font.family[0]
    for face in font.family:
        if face in pyface_family_to_kiva_family:
            family = pyface_family_to_kiva_family[face]
            break
    else:
        family = kc.DEFAULT
    size = int(font.size)
    weight = font.weight_
    style = kc.NORMAL if font.style == 'normal' else kc.ITALIC
    underline = 'underline' in font.decorations
    return Font(face_name, size, family, weight, style, underline)


class KivaFont(TraitType):
    """ A Trait which casts strings to a Kiva Font value.
    """

    #: The default value should be a tuple (factory, args, kwargs)
    default_value_type = DefaultValue.callable_and_args

    #: The parser to use when converting text to keyword args.  This should
    #: accept a string and return a dictionary of Font class trait values (ie.
    #: "family", "size", "weight", etc.).  If it can't parse the string, it
    #: should raise FontParseError.
    parser = None

    def __init__(self, default_value=None, *, parser=simple_parser, **metadata):  # noqa: E501
        self.parser = parser
        default_value = self._get_default_value(default_value)
        super().__init__(default_value, **metadata)

    def validate(self, object, name, value):
        if isinstance(value, Font):
            return value
        if isinstance(value, PyfaceFont):
            return pyface_font_to_font(value)
        if isinstance(value, str):
            try:
                return Font(**self.parser(value))
            except FontParseError:
                self.error(object, name, value)

        self.error(object, name, value)

    def info(self):
        return (
            "a Kiva Font, a Pyface Font, or a string describing a font"
        )

    def get_editor(self, trait):
        from enable.trait_defs.ui.kiva_font_editor import KivaFontEditor
        return KivaFontEditor()

    def clone(self, default_value=NoDefaultSpecified, **metadata):
        # Need to override clone due to Traits issue #1629
        new = super().clone(NoDefaultSpecified, **metadata)
        if default_value is not NoDefaultSpecified:
            new.default_value = self._get_default_value(default_value)
            new.default_value_type = DefaultValue.callable_and_args
        return new

    def _get_default_value(self, default_value):
        """Construct a default value suitable for callable_and_args."""
        if default_value is not None:
            try:
                font = self.validate(None, None, default_value)
            except TraitError:
                raise ValueError(
                    f"expected {self.info()}, but got {default_value!r}"
                )
            klass = font.__class__
            kwargs = {attr: getattr(font, attr) for attr in font_attrs}
        else:
            klass = Font
            kwargs = {}
        return (klass, (), kwargs)
