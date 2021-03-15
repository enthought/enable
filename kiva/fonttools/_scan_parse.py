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

from fontTools.afmLib import AFM
from fontTools.ttLib import TTCollection, TTFont, TTLibError

from kiva.fonttools._constants import weight_dict
from kiva.fonttools._database import FontDatabase, FontEntry
from kiva.fonttools._util import get_ttf_prop_dict, weight_as_number

logger = logging.getLogger(__name__)
# Error message when fonts fail to load
_FONT_ENTRY_ERR_MSG = "Could not convert font to FontEntry for file %s"


def create_font_database(fontfiles, fontext="ttf"):
    """ Creates a :class:`FontDatabase` instance from a list of provided
    filepaths.

    The default is to locate TrueType fonts. An AFM database can optionally be
    created.
    """
    # Use a set() to filter out files which were already scanned
    seen = set()

    fontlist = []
    for fpath in fontfiles:
        logger.debug("create_font_database %s", fpath)
        fname = os.path.basename(fpath)
        if fname in seen:
            continue

        seen.add(fname)
        if fontext == "afm":
            fontlist.extend(_build_afm_entries(fpath))
        else:
            fontlist.extend(_build_ttf_entries(fpath))

    return FontDatabase(fontlist)


def update_font_database(database, fontfiles, fontext="ttf"):
    """ Add additional font entries to an existing :class:`FontDatabase`
    instance.
    """
    fontlist = []
    for fpath in fontfiles:
        if fontext == "afm":
            fontlist.extend(_build_afm_entries(fpath))
        else:
            fontlist.extend(_build_ttf_entries(fpath))

    database.add_fonts(fontlist)


# ----------------------------------------------------------------------------
# utility funcs

def _build_afm_entries(fpath):
    """ Given the path to an AFM file, return a list of one :class:`FontEntry`
    instance or an empty list if there was an error.
    """
    try:
        font = AFM(fpath)
    except Exception:
        logger.error(f"Could not parse font file {fpath}", exc_info=True)
        return []

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
                    for idx, font in enumerate(collection.fonts):
                        entries.append(_ttf_font_property(fpath, font, idx))
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


def _afm_font_property(fpath, font):
    """ A function for populating a :class:`FontEntry` instance by
    extracting information from the AFM font file.

    *font* is a class:`AFM` instance.
    """
    family = font.FamilyName
    fontname = font.FullName.lower()

    #  Styles are: italic, oblique, and normal (default)
    if float(font.ItalicAngle) != 0.0 or family.lower().find("italic") >= 0:
        style = "italic"
    elif family.lower().find("oblique") >= 0:
        style = "oblique"
    else:
        style = "normal"

    #  Variants are: small-caps and normal (default)
    # NOTE: Not sure how many fonts actually have these strings in their family
    variant = "normal"
    for value in ("capitals", "small-caps", "smallcaps"):
        if value in family.lower():
            variant = "small-caps"
            break

    #  Weights are: 100, 200, 300, 400 (normal: default), 500 (medium),
    #    600 (semibold, demibold), 700 (bold), 800 (heavy), 900 (black)
    #    lighter and bolder are also allowed.
    weight = weight_as_number(font.Weight.lower())

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
    return FontEntry(
        fname=fpath,
        family=family,
        style=style,
        variant=variant,
        weight=weight,
        stretch=stretch,
        size=size,
    )


def _ttf_font_property(fpath, font, face_index=0):
    """ A function for populating the :class:`FontEntry` by extracting
    information from the TrueType font file.

    *font* is a :class:`TTFont` instance.
    """
    props = get_ttf_prop_dict(font)
    family = props.get("family")
    if family is None:
        raise KeyError("No family could be found for: {}".format(fpath))

    # Some properties
    full_name = props.get("full_name", "").lower()
    style_prop = props.get("style", "").lower()
    if style_prop == "":
        # For backwards compatibility with previous parsing behavior
        style_prop = full_name

    #  Styles are: italic, oblique, and normal (default)
    if style_prop.find("oblique") >= 0:
        style = "oblique"
    elif style_prop.find("italic") >= 0:
        style = "italic"
    else:
        style = "normal"

    #  Variants are: small-caps and normal (default)
    # NOTE: Not sure how many fonts actually have these strings in their family
    variant = "normal"
    for value in ("capitals", "small-caps", "smallcaps"):
        if value in family.lower():
            variant = "small-caps"
            break

    #  Weights are: 100, 200, 300, 400 (normal: default), 500 (medium),
    #    600 (semibold, demibold), 700 (bold), 800 (heavy), 900 (black)
    #    lighter and bolder are also allowed.
    weight = None
    for w in weight_dict.keys():
        if style_prop.find(w) >= 0:
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
    if full_name.find("demi cond") >= 0:
        stretch = "semi-condensed"
    elif (full_name.find("narrow") >= 0
            or full_name.find("condensed") >= 0
            or full_name.find("cond") >= 0):
        stretch = "condensed"
    elif full_name.find("wide") >= 0 or full_name.find("expanded") >= 0:
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
    return FontEntry(
        fname=fpath,
        family=family,
        style=style,
        variant=variant,
        weight=weight,
        stretch=stretch,
        size=size,
        face_index=face_index,
    )
