# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import warnings

from traits.etsconfig.api import ETSConfig

from kiva.fonttools.font_manager import default_font_manager


def add_application_font(filename):
    """ Add a TrueType font to the system in a way that makes it available to
    both the GUI toolkit and Kiva.

    Parameters
    ----------
    filename : str
        The filesystem path of a TrueType or OpenType font file.
    """
    # Handle Kiva
    fm = default_font_manager()
    fm.update_fonts([filename])

    # Handle the GUI toolkit
    if ETSConfig.toolkit.startswith("qt"):
        _qt_impl(filename)
    elif ETSConfig.toolkit == "wx":
        _wx_impl(filename)


def _qt_impl(filename):
    from pyface.qt import QtGui

    QtGui.QFontDatabase().addApplicationFont(filename)


def _wx_impl(filename):
    import wx

    if hasattr(wx.Font, "CanUsePrivateFont") and wx.Font.CanUsePrivateFont():
        wx.Font.AddPrivateFont(filename)
    else:
        warnings.warn("Wx does not support private fonts! Failed to add.")
