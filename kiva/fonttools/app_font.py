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


def add_application_fonts(filenames):
    """ Add a TrueType font to the system in a way that makes it available to
    both the GUI toolkit and Kiva.

    Parameters
    ----------
    filenames : list of str
        Filesystem paths of TrueType or OpenType font files.
    """
    if isinstance(filenames, str):
        filenames = [filenames]

    # Handle Kiva
    fm = default_font_manager()
    fm.update_fonts(filenames)

    # Handle the GUI toolkit
    if ETSConfig.toolkit.startswith("qt"):
        _qt_impl(filenames)
    elif ETSConfig.toolkit == "wx":
        _wx_impl(filenames)


def _qt_impl(filenames):
    from pyface.qt import QtGui

    for fname in filenames:
        QtGui.QFontDatabase.addApplicationFont(fname)


def _wx_impl(filenames):
    import wx

    if hasattr(wx.Font, "CanUsePrivateFont") and wx.Font.CanUsePrivateFont():
        for fname in filenames:
            wx.Font.AddPrivateFont(fname)
    else:
        warnings.warn("Wx does not support private fonts! Failed to add.")
