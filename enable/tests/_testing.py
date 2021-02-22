# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" This module contains utilities for testing within Enable.

This is not a public module and should not to be used outside of Enable.
"""
import unittest

from traits.etsconfig.api import ETSConfig


def is_wx():
    """ Return true if the toolkit backend is wx. """
    return ETSConfig.toolkit == "wx"


def is_qt():
    """ Return true if the toolkit backend is Qt
    (that includes Qt4 or Qt5, etc.)
    """
    return ETSConfig.toolkit.startswith("qt")


def is_null():
    """ Return true if the toolkit backend is null.
    """
    return ETSConfig.toolkit == "null"


skip_if_null = unittest.skipIf(
    is_null(), "Test not working on the 'null' backend"
)

skip_if_not_qt = unittest.skipIf(not is_qt(), "Test only for qt")


skip_if_not_wx = unittest.skipIf(not is_wx(), "Test only for wx")


def get_dialog_size(ui_control):
    """Return the size of the dialog.
    Return a tuple (width, height) with the size of the dialog in pixels.
    E.g.:
        >>> get_dialog_size(ui.control)
    """

    if is_wx():
        return ui_control.GetSize()

    elif is_qt():
        return ui_control.size().width(), ui_control.size().height()

    raise RuntimeError("Unable to compute dialog size. Unknown toolkit.")
