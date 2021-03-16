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
from kiva.fonttools._constants import (
    font_family_aliases, preferred_fonts, stretch_dict, weight_dict
)

# Each of the scoring functions below should return a value between
# 0.0 (perfect match) and 1.0 (terrible match)


def score_family(families, family2):
    """ Returns a match score between the list of font families in
    *families* and the font family name *family2*.

    An exact match anywhere in the list returns 0.0.

    A match by generic font name will return 0.1.

    No match will return 1.0.
    """
    family2 = family2.lower()
    for i, family1 in enumerate(families):
        family1 = family1.lower()
        if family1 in font_family_aliases:
            if family1 in {"sans", "sans serif", "modern"}:
                family1 = "sans-serif"
            options = preferred_fonts[family1]
            options = [x.lower() for x in options]
            if family2 in options:
                idx = options.index(family2)
                return 0.1 * (float(idx) / len(options))
        elif family1 == family2:
            return 0.0
    return 1.0


def score_size(size1, size2):
    """ Returns a match score between *size1* and *size2*.

    If *size2* (the size specified in the font file) is 'scalable', this
    function always returns 0.0, since any font size can be generated.

    Otherwise, the result is the absolute distance between *size1* and
    *size2*, normalized so that the usual range of font sizes (6pt -
    72pt) will lie between 0.0 and 1.0.
    """
    if size2 == "scalable":
        return 0.0
    # Size value should have already been
    try:
        sizeval1 = float(size1)
    except ValueError:
        return 1.0
    try:
        sizeval2 = float(size2)
    except ValueError:
        return 1.0
    return abs(sizeval1 - sizeval2) / 72.0


def score_stretch(stretch1, stretch2):
    """ Returns a match score between *stretch1* and *stretch2*.

    The result is the absolute value of the difference between the
    CSS numeric values of *stretch1* and *stretch2*, normalized
    between 0.0 and 1.0.
    """
    try:
        stretchval1 = int(stretch1)
    except ValueError:
        stretchval1 = stretch_dict.get(stretch1, 500)
    try:
        stretchval2 = int(stretch2)
    except ValueError:
        stretchval2 = stretch_dict.get(stretch2, 500)
    return abs(stretchval1 - stretchval2) / 1000.0


def score_style(style1, style2):
    """ Returns a match score between *style1* and *style2*.

    * An exact match returns 0.0.
    * A match between 'italic' and 'oblique' returns 0.1.
    * No match returns 1.0.
    """
    styles = ("italic", "oblique")
    if style1 == style2:
        return 0.0
    elif style1 in styles and style2 in styles:
        return 0.1
    return 1.0


def score_variant(variant1, variant2):
    """ Returns a match score between *variant1* and *variant2*.

    An exact match returns 0.0, otherwise 1.0.
    """
    if variant1 == variant2:
        return 0.0
    else:
        return 1.0


def score_weight(weight1, weight2):
    """ Returns a match score between *weight1* and *weight2*.

    The result is the absolute value of the difference between the
    CSS numeric values of *weight1* and *weight2*, normalized
    between 0.0 and 1.0.
    """
    try:
        weightval1 = int(weight1)
    except ValueError:
        weightval1 = weight_dict.get(weight1, 500)
    try:
        weightval2 = int(weight2)
    except ValueError:
        weightval2 = weight_dict.get(weight2, 500)
    return abs(weightval1 - weightval2) / 1000.0
