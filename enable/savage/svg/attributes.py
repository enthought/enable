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
    Parsers for specific attributes
"""
import urllib.parse as urlparse

from pyparsing import (
    CaselessLiteral, Group, Literal, Optional, SkipTo, StringEnd
)

from .css.colour import colourValue

# Paint values
none = CaselessLiteral("none").setParseAction(lambda t: ["NONE", ()])
currentColor = CaselessLiteral("currentColor").setParseAction(
    lambda t: ["CURRENTCOLOR", ()]
)


def parsePossibleURL(t):
    # Workaround for PyParsing versions < 2.1.0, for which t is wrapped in an
    # extra level of nesting. See enthought/enable#224.
    if len(t) == 1:
        t = t[0]

    possibleURL, fallback = t
    return [urlparse.urlsplit(possibleURL), fallback]


# Normal color declaration
colorDeclaration = none | currentColor | colourValue

urlEnd = (
    Literal(")").suppress()
    + Optional(Group(colorDeclaration), default=())
    + StringEnd()
)

url = (
    CaselessLiteral("URL")
    + Literal("(").suppress()
    + Group(SkipTo(urlEnd, include=True).setParseAction(parsePossibleURL))
)

# paint value will parse into a (type, details) tuple.
# For none and currentColor, the details tuple will be the empty tuple
# for CSS color declarations, it will be (type, (R,G,B))
# for URLs, it will be ("URL", ((url tuple), fallback))
# The url tuple will be as returned by urlparse.urlsplit, and can be
# an empty tuple if the parser has an error
# The fallback will be another (type, details) tuple as a parsed
# colorDeclaration, but may be the empty tuple if it is not present
paintValue = url | colorDeclaration
