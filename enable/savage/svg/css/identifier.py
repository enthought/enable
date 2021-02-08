# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Parse CSS identifiers. More complicated than it sounds"""
from pyparsing import Combine, Literal, Optional, Regex, White, ZeroOrMore
import re


class White(White):
    """ Customize whitespace to match the CSS spec values"""

    def __init__(self, ws=" \t\r\n\f", min=1, max=0, exact=0):
        super(White, self).__init__(ws, min, max, exact)


escaped = (
    Literal("\\").suppress()
    +
    # chr(20)-chr(126) + chr(128)-unichr(sys.maxunicode)
    Regex("[\u0020-\u007e\u0080-\uffff]", re.IGNORECASE)
)


def convertToUnicode(t):
    return chr(int(t[0], 16))


hex_unicode = (
    Literal("\\").suppress()
    + Regex("[0-9a-f]{1,6}", re.IGNORECASE)
    + Optional(White(exact=1)).suppress()
).setParseAction(convertToUnicode)


escape = hex_unicode | escaped

# any unicode literal outside the 0-127 ascii range
nonascii = Regex("[^\u0000-\u007f]")

# single character for starting an identifier.
nmstart = Regex("[A-Z]", re.IGNORECASE) | nonascii | escape

nmchar = Regex("[0-9A-Z-]", re.IGNORECASE) | nonascii | escape

identifier = Combine(nmstart + ZeroOrMore(nmchar))
