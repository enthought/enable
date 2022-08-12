# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the Kiva Font class and a utility method to parse free-form font
specification strings into Font instances.
"""
import copy
import warnings

from kiva.constants import (
    BOLD, DECORATIVE, DEFAULT, ITALIC, MODERN, NORMAL, ROMAN, SCRIPT, SWISS,
    TELETYPE, WEIGHT_BOLD, WEIGHT_EXTRABOLD, WEIGHT_EXTRAHEAVY,
    WEIGHT_EXTRALIGHT, WEIGHT_HEAVY, WEIGHT_LIGHT, WEIGHT_MEDIUM,
    WEIGHT_NORMAL, WEIGHT_SEMIBOLD, WEIGHT_THIN, bold_styles, italic_styles,
)
from kiva.fonttools._query import FontQuery
from kiva.fonttools.font_manager import default_font_manager

FAMILIES = {
    'default': DEFAULT,
    'cursive': SCRIPT,
    'decorative': DECORATIVE,
    'fantasy': DECORATIVE,
    'modern': MODERN,
    'monospace': MODERN,
    'roman': ROMAN,
    'sans-serif': SWISS,
    'script': SCRIPT,
    'serif': ROMAN,
    'swiss': SWISS,
    'teletype': TELETYPE,
    'typewriter': TELETYPE,
}
WEIGHTS = {
    'thin': WEIGHT_THIN,
    'extra-light': WEIGHT_EXTRALIGHT,
    'light': WEIGHT_LIGHT,
    'regular': WEIGHT_NORMAL,
    'medium': WEIGHT_MEDIUM,
    'semi-bold': WEIGHT_SEMIBOLD,
    'bold': WEIGHT_BOLD,
    'extra-bold': WEIGHT_EXTRABOLD,
    'heavy': WEIGHT_HEAVY,
    'extra-heavy': WEIGHT_EXTRAHEAVY
}
STYLES = {
    'italic': ITALIC,
    'oblique': ITALIC,
}
DECORATIONS = {'underline'}
NOISE = {'pt', 'point', 'px', 'family'}


class FontParseError(ValueError):
    """An exception raised when font parsing fails."""
    pass


def simple_parser(description):
    """An extremely simple font description parser.

    The parser is simple, and works by splitting the description on whitespace
    and examining each resulting token for understood terms:

    Size
        The first numeric term is treated as the font size.

    Weight
        The following weight terms are accepted: 'thin', 'extra-light',
        'light', 'regular', 'medium', 'semi-bold', 'bold', 'extra-bold',
        'heavy', 'extra-heavy'.

    Style
        The following style terms are accepted: 'italic', 'oblique'.

    Decorations
        The following decoration term is accepted: 'underline'

    Generic Families
        The following generic family terms are accepted: 'default', 'cursive',
        'decorative', 'fantasy', 'modern', 'monospace', 'roman', 'sans-serif',
        'script', 'serif', 'swiss', 'teletype', 'typewriter'.

    In addition, the parser ignores the terms 'pt', 'point', 'px', and 'family'.
    Any remaining terms are combined into the typeface name.  There is no
    expected order to the terms.

    This parser is roughly compatible with the various ad-hoc parsers in
    TraitsUI and Kiva, allowing for the slight differences between them and
    adding support for additional options supported by Pyface fonts, such as
    stretch and variants.

    Parameters
    ----------
    description : str
        The font description to be parsed.

    Returns
    -------
    properties : dict
        Font properties suitable for use in creating a Pyface Font.

    Notes
    -----
    This is not a particularly good parser, as it will fail to properly
    parse something like "10 pt times new roman" or "14 pt computer modern"
    since they have generic font names as part of the font face name.

    This is derived from Pyface's equivalent simple_parser.  Eventually both
    will be replaced by better parsers that can parse something closer to a
    CSS font definition.
    """
    face = []
    family = DEFAULT
    size = None
    weight = WEIGHT_NORMAL
    style = NORMAL
    underline = False
    for word in description.split():
        lower_word = word.casefold()
        if lower_word in NOISE:
            continue
        elif lower_word in FAMILIES:
            family = FAMILIES[lower_word]
        elif lower_word in WEIGHTS:
            weight = WEIGHTS[lower_word]
        elif lower_word in STYLES:
            style = STYLES[lower_word]
        elif lower_word in DECORATIONS:
            underline = True
        else:
            if size is None:
                try:
                    size = int(lower_word)
                    continue
                except ValueError:
                    pass
            face.append(word)

    face_name = " ".join(face)
    if size is None:
        size = 10

    return {
        'face_name': face_name,
        'size': size,
        'family': family,
        'weight': weight,
        'style': style,
        'underline': underline,
    }


def str_to_font(fontspec, parser=simple_parser):
    """
    Converts a string specification of a font into a Font instance.
    string specifications are of the form: "modern 12", "9 roman italic",
    and so on.
    """
    font_properties = parser(fontspec)
    return Font(**font_properties)


class Font(object):
    """ Font class for device independent font specification.

        It is primarily based on wxPython, but looks to be similar to
        the needs of Mac OS X, etc.

        The family defaults to SWISS so that font rotation will work
        correctly under wxPython.    Revisit as we get more platforms
        defined.
    """

    # Maps the constants for font families to names to use when searching for
    # fonts.
    familymap = {
        DEFAULT: "serif",
        SWISS: "sans-serif",
        ROMAN: "serif",
        MODERN: "sans-serif",
        DECORATIVE: "fantasy",
        SCRIPT: "cursive",
        TELETYPE: "monospace",
    }

    def __init__(self, face_name="", size=12, family=SWISS,
                 weight=WEIGHT_NORMAL, style=NORMAL, underline=0,
                 encoding=DEFAULT):
        if not isinstance(face_name, str):
            raise RuntimeError(
                f"Expected face name to be a str, got {face_name!r}")
        if not isinstance(size, int):
            raise RuntimeError(
                f"Expected size to be an int, got {size!r}")
        if not isinstance(family, int):
            raise RuntimeError(
                f"Expected family to be an int, got {family!r}")
        if not isinstance(weight, int):
            raise RuntimeError(
                f"Expected weight to be an int, got {weight!r}")
        if not isinstance(style, int):
            raise RuntimeError(
                f"Expected style to be an int, got {style!r}")
        if not isinstance(underline, int):
            raise RuntimeError(
                f"Expected underline to be a int, got {underline!r}")
        if not isinstance(encoding, int):
            raise RuntimeError(
                f"Expected encoding to be an int, got {encoding!r}")

        self.face_name = face_name
        self.size = size
        self.family = family
        self.weight = weight
        self.style = style
        self.underline = bool(underline)
        self.encoding = encoding

        # correct the style and weight if needed (can be removed in Enable 7)
        self.weight = self._get_weight()
        self.style = style & ~BOLD

    def findfont(self, language=None):
        """ Returns the file name and face index of the font that most closely
        matches our font properties.

        Parameters
        ----------
        language : str [optional]
            If provided, attempt to find a font which supports ``language``.
        """
        query = self._make_font_query()
        if language is not None:
            spec = default_font_manager().find_fallback(query, language)
            if spec is not None:
                return spec

        return default_font_manager().findfont(query)

    def findfontname(self, language=None):
        """ Returns the name of the font that most closely matches our font
        properties.

        Parameters
        ----------
        language : str [optional]
            If provided, attempt to find a font which supports ``language``.
        """
        query = self._make_font_query()
        if language is not None:
            spec = default_font_manager().find_fallback(query, language)
            if spec is not None:
                return spec.family

        return query.get_name()

    def is_bold(self):
        """Is the font considered bold or not?

        This is a convenience method for backends which don't fully support
        font weights.  We consider a font to be bold if its weight is more
        than medium.
        """
        weight = self._get_weight()
        return (weight > WEIGHT_MEDIUM)

    def _make_font_query(self):
        """ Returns a FontQuery object that encapsulates our font properties.
        """
        weight = self._get_weight()

        if self.style in italic_styles:
            style = "italic"
        else:
            style = "normal"

        query = FontQuery(
            family=self.familymap[self.family],
            style=style,
            weight=weight,
            size=self.size,
        )
        if self.face_name != "":
            query.set_name(self.face_name)
        return query

    def _get_weight(self):
        """Get a corrected weight value from the font.

        Note: this is a temporary method that will be removed in Enable 7.
        """
        if self.weight == BOLD:
            warnings.warn(
                "Use WEIGHT_BOLD instead of BOLD for Font.weight",
                DeprecationWarning
            )
            return WEIGHT_BOLD
        elif self.style in bold_styles:
            warnings.warn(
                "Set Font.weight to WEIGHT_BOLD instead of Font.style to "
                "BOLD or BOLD_STYLE",
                DeprecationWarning
            )
            # if weight is default, and style is bold, report as bold
            if self.weight == WEIGHT_NORMAL:
                return WEIGHT_BOLD

        return self.weight

    def _get_name(self):
        return self.face_name

    def _set_name(self, val):
        self.face_name = val

    name = property(_get_name, _set_name)

    def copy(self):
        """ Returns a copy of the font object.
        """
        return copy.deepcopy(self)

    def __eq__(self, other):
        try:
            return (self.family == other.family
                    and self.face_name == other.face_name
                    and self.size == other.size
                    and self.weight == other.weight
                    and self.style == other.style
                    and self.underline == other.underline
                    and self.encoding == other.encoding)
        except AttributeError:
            pass
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return (
            f"Font(face_name='{self.face_name}', size={self.size}, "
            f"family={self.family}, weight={self.weight}, style={self.style}, "
            f"underline={self.underline}, encoding={self.encoding})"
        )
