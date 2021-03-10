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
####### NOTE #######
This is based heavily on matplotlib's font_manager.py SVN rev 8713
(git commit f8e4c6ce2408044bc89b78b3c72e54deb1999fb5),
but has been modified quite a bit in the decade since it was copied.
####################
"""
import logging
import os

from fontTools.ttLib import TTCollection, TTFont, TTLibError

from kiva.fonttools import afm
from kiva.fonttools._constants import weight_dict
from kiva.fonttools._util import get_ttf_prop_dict, weight_as_number

logger = logging.getLogger(__name__)
# Error message when fonts fail to load
_FONT_ENTRY_ERR_MSG = "Could not convert font to FontEntry for file %s"


def create_font_list(fontfiles, fontext="ttf"):
    """ Creates a list of :class`FontEntry` instances from a list of provided
    filepaths.

    The default is to create a list of TrueType fonts. An AFM font list can
    optionally be created.
    """
    # Use a set() to filter out files which were already scanned
    seen = set()

    fontlist = []
    for fpath in fontfiles:
        logger.debug("create_font_list %s", fpath)
        fname = os.path.basename(fpath)
        if fname in seen:
            continue

        seen.add(fname)
        if fontext == "afm":
            fontlist.extend(_build_afm_entries(fpath))
        else:
            fontlist.extend(_build_ttf_entries(fpath))

    return fontlist


class FontEntry(object):
    """ A class for storing Font properties. It is used when populating
    the font lookup dictionary.
    """
    def __init__(self, fname="", name="", style="normal", variant="normal",
                 weight="normal", stretch="normal", size="medium"):
        self.fname = fname
        self.name = name
        self.style = style
        self.variant = variant
        self.weight = weight
        self.stretch = stretch
        try:
            self.size = str(float(size))
        except ValueError:
            self.size = size

    def __repr__(self):
        fname = os.path.basename(self.fname)
        return (
            f"<FontEntry '{self.name}' ({fname}) {self.style} {self.variant} "
            f"{self.weight} {self.stretch}>"
        )


# ----------------------------------------------------------------------------
# utility funcs

def _build_afm_entries(fpath):
    """ Given the path to an AFM file, return a list of one :class:`FontEntry`
    instance or an empty list if there was an error.
    """
    try:
        fh = open(fpath, "r")
    except OSError:
        logger.error(f"Could not open font file {fpath}", exc_info=True)
        return []

    try:
        font = afm.AFM(fh)
    except RuntimeError:
        logger.error(f"Could not parse font file {fpath}", exc_info=True)
        return []
    finally:
        fh.close()

    try:
        return [_afm_font_property(fpath, font)]
    except Exception:
        logger.error(_FONT_ENTRY_ERR_MSG, fpath, exc_info=True)

    return []


def _build_ttf_entries(fpath):
    """ Given the path to a TTF/TTC file, return a list of :class:`FontEntry`
    instances.
    """
    entries = []

    ext = os.path.splitext(fpath)[-1]
    try:
        with open(fpath, "rb") as fp:
            if ext.lower() == ".ttc":
                collection = TTCollection(fp)
                try:
                    for font in collection.fonts:
                        entries.append(_ttf_font_property(fpath, font))
                except Exception:
                    logger.error(_FONT_ENTRY_ERR_MSG, fpath, exc_info=True)
            else:
                font = TTFont(fp)
                try:
                    entries.append(_ttf_font_property(fpath, font))
                except Exception:
                    logger.error(_FONT_ENTRY_ERR_MSG, fpath, exc_info=True)
    except (RuntimeError, TTLibError):
        logger.error(f"Could not open font file {fpath}", exc_info=True)
    except UnicodeError:
        logger.error(f"Cannot handle unicode file: {fpath}", exc_info=True)

    return entries


def _afm_font_property(fontpath, font):
    """ A function for populating a :class:`FontEntry` instance by
    extracting information from the AFM font file.

    *font* is a class:`AFM` instance.
    """
    name = font.get_familyname()
    fontname = font.get_fontname().lower()

    #  Styles are: italic, oblique, and normal (default)
    if font.get_angle() != 0 or name.lower().find("italic") >= 0:
        style = "italic"
    elif name.lower().find("oblique") >= 0:
        style = "oblique"
    else:
        style = "normal"

    #  Variants are: small-caps and normal (default)
    # NOTE: Not sure how many fonts actually have these strings in their name
    variant = "normal"
    for value in ("capitals", "small-caps"):
        if value in name.lower():
            variant = "small-caps"
            break

    #  Weights are: 100, 200, 300, 400 (normal: default), 500 (medium),
    #    600 (semibold, demibold), 700 (bold), 800 (heavy), 900 (black)
    #    lighter and bolder are also allowed.
    weight = weight_as_number(font.get_weight().lower())

    #  Stretch can be absolute and relative
    #  Absolute stretches are: ultra-condensed, extra-condensed, condensed,
    #    semi-condensed, normal, semi-expanded, expanded, extra-expanded,
    #    and ultra-expanded.
    #  Relative stretches are: wider, narrower
    #  Child value is: inherit
    if fontname.find("demi cond") >= 0:
        stretch = "semi-condensed"
    elif (fontname.find("narrow") >= 0
            or fontname.find("condensed") >= 0
            or fontname.find("cond") >= 0):
        stretch = "condensed"
    elif fontname.find("wide") >= 0 or fontname.find("expanded") >= 0:
        stretch = "expanded"
    else:
        stretch = "normal"

    #  Sizes can be absolute and relative.
    #  Absolute sizes are: xx-small, x-small, small, medium, large, x-large,
    #    and xx-large.
    #  Relative sizes are: larger, smaller
    #  Length value is an absolute font size, e.g. 12pt
    #  Percentage values are in 'em's.  Most robust specification.

    #  All AFM fonts are apparently scalable.
    size = "scalable"
    return FontEntry(fontpath, name, style, variant, weight, stretch, size)


def _ttf_font_property(fpath, font):
    """ A function for populating the :class:`FontEntry` by extracting
    information from the TrueType font file.

    *font* is a :class:`TTFont` instance.
    """
    props = get_ttf_prop_dict(font)
    name = props.get("name")
    if name is None:
        raise KeyError("No name could be found for: {}".format(fpath))

    #  Styles are: italic, oblique, and normal (default)
    sfnt4 = props.get("sfnt4", "").lower()
    if sfnt4.find("oblique") >= 0:
        style = "oblique"
    elif sfnt4.find("italic") >= 0:
        style = "italic"
    else:
        style = "normal"

    #  Variants are: small-caps and normal (default)
    # NOTE: Not sure how many fonts actually have these strings in their name
    variant = "normal"
    for value in ("capitals", "small-caps"):
        if value in name.lower():
            variant = "small-caps"
            break

    #  Weights are: 100, 200, 300, 400 (normal: default), 500 (medium),
    #    600 (semibold, demibold), 700 (bold), 800 (heavy), 900 (black)
    #    lighter and bolder are also allowed.
    weight = None
    for w in weight_dict.keys():
        if sfnt4.find(w) >= 0:
            weight = w
            break
    if not weight:
        weight = 400
    weight = weight_as_number(weight)

    #  Stretch can be absolute and relative
    #  Absolute stretches are: ultra-condensed, extra-condensed, condensed,
    #    semi-condensed, normal, semi-expanded, expanded, extra-expanded,
    #    and ultra-expanded.
    #  Relative stretches are: wider, narrower
    #  Child value is: inherit
    if sfnt4.find("demi cond") >= 0:
        stretch = "semi-condensed"
    elif (sfnt4.find("narrow") >= 0
            or sfnt4.find("condensed") >= 0
            or sfnt4.find("cond") >= 0):
        stretch = "condensed"
    elif sfnt4.find("wide") >= 0 or sfnt4.find("expanded") >= 0:
        stretch = "expanded"
    else:
        stretch = "normal"

    #  Sizes can be absolute and relative.
    #  Absolute sizes are: xx-small, x-small, small, medium, large, x-large,
    #    and xx-large.
    #  Relative sizes are: larger, smaller
    #  Length value is an absolute font size, e.g. 12pt
    #  Percentage values are in 'em's.  Most robust specification.

    #  !!!!  Incomplete
    size = "scalable"
    return FontEntry(fpath, name, style, variant, weight, stretch, size)
