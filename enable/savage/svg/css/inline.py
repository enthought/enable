# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Parser for inline CSS in style attributes """


def inlineStyle(styleString):
    if len(styleString) == 0:
        return {}
    styles = styleString.split(";")
    rv = dict(style.split(":") for style in styles if len(style) != 0)
    return rv
