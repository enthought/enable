# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Parser for various kinds of CSS values as per CSS2 spec section 4.3
"""
from pyparsing import Combine, Literal, Optional, StringEnd, Word, oneOf


def asInt(s, l, t):
    return int(t[0])


def asFloat(s, l, t):
    return float(t[0])


def asFloatOrInt(s, l, t):
    """ Return an int if possible, otherwise a float"""
    v = t[0]
    try:
        return int(v)
    except ValueError:
        return float(v)


integer = Word("0123456789").setParseAction(asInt)

number = Combine(
    Optional(Word("0123456789")) + Literal(".") + Word("01234567890") | integer
)
number.setName("number")


sign = oneOf("+ -")

signedNumber = Combine(Optional(sign) + number).setParseAction(asFloat)

lengthValue = Combine(Optional(sign) + number).setParseAction(asFloatOrInt)
lengthValue.setName("lengthValue")


lengthUnit = oneOf(
    ["em", "ex", "px", "pt", "in", "cm", "mm", "pc", "%"], caseless=True
)
# the spec says that the unit is only optional for a 0 length, but
# there are just too many places where a default is permitted.
# TODO: Maybe should use a ctor like optional to let clients declare it?
length = lengthValue + Optional(lengthUnit, default=None) + StringEnd()
length.leaveWhitespace()

# set the parse action aftward so it doesn't "infect" the parsers that build
# on it
number.setParseAction(asFloat)
