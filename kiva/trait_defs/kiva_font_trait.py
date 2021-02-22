# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
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

# -----------------------------------------------------------------------------
#  Imports:
# -----------------------------------------------------------------------------

from traits.etsconfig.api import ETSConfig

if ETSConfig.toolkit == "wx":
    from .ui.wx.kiva_font_editor import KivaFontEditor
elif ETSConfig.toolkit.startswith("qt"):
    # FIXME
    # from .ui.qt4.kiva_font_editor import KivaFontEditor
    KivaFontEditor = None
else:
    KivaFontEditor = None

from traits.api import Trait, TraitError, TraitHandler, TraitFactory


# -----------------------------------------------------------------------------
#  Convert a string into a valid 'Font' object (if possible):
# -----------------------------------------------------------------------------

# Strings to ignore in text representations of fonts
font_noise = ["pt", "point", "family"]

font_families = font_styles = font_weights = DEFAULT = NORMAL = None


def init_constants():
    """ Dynamically load Kiva constants to avoid import dependencies.
    """
    global font_families, font_styles, font_weights, default_face
    global DEFAULT, NORMAL

    if font_families is not None:
        return

    import kiva.constants as kc

    DEFAULT = kc.DEFAULT
    NORMAL = kc.NORMAL

    # Mapping of strings to valid Kiva font families:
    font_families = {
        "default": kc.DEFAULT,
        "decorative": kc.DECORATIVE,
        "roman": kc.ROMAN,
        "script": kc.SCRIPT,
        "swiss": kc.SWISS,
        "modern": kc.MODERN,
    }

    # Mapping of strings to Kiva font styles:
    font_styles = {"italic": kc.ITALIC}

    # Mapping of strings to Kiva font weights:
    font_weights = {"bold": kc.BOLD}

    default_face = {
        kc.SWISS: "Arial",
        kc.ROMAN: "Times",
        kc.MODERN: "Courier",
        kc.SCRIPT: "Zapfino",
        kc.DECORATIVE: "Zapfino",  # need better choice for this
    }


# Strings to ignore in text representations of fonts
font_noise = ["pt", "point", "family"]

# -----------------------------------------------------------------------------
#  'TraitKivaFont' class'
# -----------------------------------------------------------------------------


class TraitKivaFont(TraitHandler):
    """ Ensures that values assigned to a trait attribute are valid font
    descriptor strings for Kiva fonts; the value actually assigned is the
    corresponding Kiva font.
    """

    # -------------------------------------------------------------------------
    #  Validates that the value is a valid font:
    # -------------------------------------------------------------------------

    def validate(self, object, name, value):
        """ Validates that the value is a valid font.
        """
        from kiva.fonttools import Font

        if isinstance(value, Font):
            return value

        # Make sure all Kiva related data is loaded:
        init_constants()

        try:
            point_size = 10
            family = DEFAULT
            style = NORMAL
            weight = NORMAL
            underline = 0
            facename = []
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
                        facename.append(word)

            if facename == "":
                facename = default_face.get(family, "")
            # FIXME: The above if clause never happens, the below should
            # be correct though it results in loading weird fonts.
            #             if facename == []:
            #                 facename = [default_face.get(family, "")]
            return Font(
                face_name=" ".join(facename),
                size=point_size,
                family=family,
                weight=weight,
                style=style,
                underline=underline,
            )
        except Exception:
            pass

        raise TraitError(object, name, "a font descriptor string", repr(value))

    def info(self):
        return (
            "a string describing a font (e.g. '12 pt bold italic "
            "swiss family Arial' or 'default 12')"
        )


fh = TraitKivaFont()
if KivaFontEditor is not None:
    KivaFontTrait = Trait(
        fh.validate(None, None, "modern 12"), fh, editor=KivaFontEditor
    )
else:
    KivaFontTrait = Trait(fh.validate(None, None, "modern 12"), fh)


def KivaFontFunc(*args, **metadata):
    """ Returns a trait whose value must be a GUI toolkit-specific font.

    Description
    -----------
    For wxPython, the returned trait accepts any of the following:

    * an kiva.fonttools.Font instance
    * a string describing the font, including one or more of the font family,
      size, weight, style, and typeface name.

    Default Value
    -------------
    For wxPython, 'Arial 10'
    """
    return KivaFontTrait(*args, **metadata)


KivaFont = TraitFactory(KivaFontFunc)
