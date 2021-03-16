# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from fontTools.afmLib import AFM
from fontTools.ttLib import TTFont

from kiva.fonttools._constants import stretch_dict, weight_dict
from kiva.fonttools._util import get_ttf_prop_dict
from kiva.fonttools.font_manager import default_font_manager


class FontQuery(object):
    """ A class for storing properties needed to query the font manager.

    The properties are those described in the `W3C Cascading
    Style Sheet, Level 1 <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ font
    specification. The six properties are:

      - family: A list of font names in decreasing order of priority.
        The items may include a generic font family name, either
        'serif', 'sans-serif', 'cursive', 'fantasy', or 'monospace'.

      - style: Either 'normal', 'italic' or 'oblique'.

      - variant: Either 'normal' or 'small-caps'.

      - stretch: A numeric value in the range 0-1000 or one of
        'ultra-condensed', 'extra-condensed', 'condensed',
        'semi-condensed', 'normal', 'semi-expanded', 'expanded',
        'extra-expanded' or 'ultra-expanded'

      - weight: A numeric value in the range 0-1000 or one of
        'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
        'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
        'extra bold', 'black'

      - size: An absolute font size, e.g. 12

    Alternatively, a font may be specified using an absolute path to a
    .ttf file, by using the *fname* kwarg.
    """
    def __init__(self, family=None, style=None, variant=None, weight=None,
                 stretch=None, size=None, fname=None, _init=None):
        # if fname is set, it's a hardcoded filename to use
        # _init is used only by copy()

        self._family = None
        self._slant = None
        self._variant = None
        self._weight = None
        self._stretch = None
        self._size = None
        self._file = None

        # This is used only by copy()
        if _init is not None:
            self.__dict__.update(_init.__dict__)
            return

        self.set_family(family)
        self.set_style(style)
        self.set_variant(variant)
        self.set_weight(weight)
        self.set_stretch(stretch)
        self.set_file(fname)
        self.set_size(size)

    def __hash__(self):
        lst = [(k, getattr(self, "get" + k)()) for k in sorted(self.__dict__)]
        return hash(repr(lst))

    def __str__(self):
        attrs = (
            self._family, self._slant, self._variant, self._weight,
            self._stretch, self._size,
        )
        return str(attrs)

    def get_family(self):
        """ Return a list of font names that comprise the font family.
        """
        return self._family

    def get_name(self):
        """ Return the name of the font that best matches the font properties.
        """
        spec = default_font_manager().findfont(self)
        if spec.filename.endswith(".afm"):
            return AFM().FamilyName

        prop_dict = get_ttf_prop_dict(
            TTFont(spec.filename, fontNumber=spec.face_index)
        )
        return prop_dict["family"]

    def get_style(self):
        """ Return the font style.

        Values are: 'normal', 'italic' or 'oblique'.
        """
        return self._slant

    get_slant = get_style

    def get_variant(self):
        """ Return the font variant.

        Values are: 'normal' or 'small-caps'.
        """
        return self._variant

    def get_weight(self):
        """ Set the font weight.

        Options are: A numeric value in the range 0-1000 or one of 'light',
        'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold',
        'demi', 'bold', 'heavy', 'extra bold', 'black'
        """
        return self._weight

    def get_stretch(self):
        """ Return the font stretch or width.

        Options are: 'ultra-condensed', 'extra-condensed', 'condensed',
        'semi-condensed', 'normal', 'semi-expanded', 'expanded',
        'extra-expanded', 'ultra-expanded'.
        """
        return self._stretch

    def get_size(self):
        """ Return the font size.
        """
        return self._size

    def get_file(self):
        """ Return the filename of the associated font.
        """
        return self._file

    def set_family(self, family):
        """ Change the font family.

        May be either an alias (generic name is CSS parlance), such as:
        'serif', 'sans-serif', 'cursive', 'fantasy', or 'monospace', or
        a real font name.
        """
        if family is None:
            self._family = None
        else:
            if isinstance(family, bytes):
                family = [family.decode("utf8")]
            elif isinstance(family, str):
                family = [family]
            self._family = family

    set_name = set_family

    def set_style(self, style):
        """ Set the font style.

        Values are: 'normal', 'italic' or 'oblique'.
        """
        if style not in ("normal", "italic", "oblique", None):
            raise ValueError("style must be normal, italic or oblique")
        self._slant = style

    set_slant = set_style

    def set_variant(self, variant):
        """ Set the font variant.

        Values are: 'normal' or 'small-caps'.
        """
        if variant not in ("normal", "small-caps", None):
            raise ValueError("variant must be normal or small-caps")
        self._variant = variant

    def set_weight(self, weight):
        """ Set the font weight.

        May be either a numeric value in the range 0-1000 or one of
        'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman',
        'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'.
        """
        if weight is not None:
            try:
                weight = int(weight)
                if weight < 0 or weight > 1000:
                    raise ValueError()
            except ValueError:
                if weight not in weight_dict:
                    raise ValueError("weight is invalid")
        self._weight = weight

    def set_stretch(self, stretch):
        """ Set the font stretch or width.

        Options are: 'ultra-condensed', 'extra-condensed', 'condensed',
        'semi-condensed', 'normal', 'semi-expanded', 'expanded',
        'extra-expanded' or 'ultra-expanded', or a numeric value in the
        range 0-1000.
        """
        if stretch is not None:
            try:
                stretch = int(stretch)
                if stretch < 0 or stretch > 1000:
                    raise ValueError()
            except ValueError:
                if stretch not in stretch_dict:
                    raise ValueError("stretch is invalid")
        else:
            stretch = 500
        self._stretch = stretch

    def set_size(self, size):
        """ Set the font size.

        An absolute font size, e.g. 12.
        """
        if size is not None:
            try:
                size = float(size)
            except ValueError:
                raise ValueError("size is invalid")
        self._size = size

    def set_file(self, file):
        """ Set the filename of the fontfile to use.

        In this case, all other properties will be ignored.
        """
        self._file = file

    def copy(self):
        """ Return a deep copy of self
        """
        return FontQuery(_init=self)
