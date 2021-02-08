# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Parsing for CSS and CSS-style values, such as transform and filter
attributes.
"""

from pyparsing import Group, Literal, Optional, delimitedList

# some shared definitions from pathdata

from enable.savage.svg.pathdata import number, maybeComma

paren = Literal("(").suppress()
cparen = Literal(")").suppress()


def Parenthised(exp):
    return Group(paren + exp + cparen)


skewY = Literal("skewY") + Parenthised(number)

skewX = Literal("skewX") + Parenthised(number)

rotate = Literal("rotate") + Parenthised(
    number + Optional(maybeComma + number + maybeComma + number)
)


scale = Literal("scale") + Parenthised(number + Optional(maybeComma + number))

translate = Literal("translate") + Parenthised(
    number + Optional(maybeComma + number)
)

matrix = Literal("matrix") + Parenthised(
    # there's got to be a better way to write this
    number
    + maybeComma
    + number
    + maybeComma
    + number
    + maybeComma
    + number
    + maybeComma
    + number
    + maybeComma
    + number
)

transform = skewY | skewX | rotate | scale | translate | matrix

transformList = delimitedList(Group(transform), delim=maybeComma)
