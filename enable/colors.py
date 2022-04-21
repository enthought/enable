# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
# This is a redirection file that determines what constitutes a color trait
# in Chaco, and what constitutes the standard colors.

from pyface.toolkit import toolkit

from enable.trait_defs.rgba_color_trait import (  # noqa: F401
    ColorTrait, black_color_trait, color_table,
    convert_to_color_tuple as convert_to_color,
    transparent_color_trait, white_color_trait
)
from enable.trait_defs.ui.rgba_color_editor import (
    RGBAColorEditor as ColorEditorFactory,
)


transparent_color = color_table["transparent"]


if toolkit.toolkit == "wx":
    from traitsui.wx.constants import WindowColor

    color_table["syswindow"] = (
        WindowColor.Red() / 255.0,
        WindowColor.Green() / 255.0,
        WindowColor.Blue() / 255.0,
        1.0,
    )
elif toolkit.toolkit.startswith("qt"):
    from pyface.qt import QtGui

    # XXX to-do: this will not work if dark mode changes (see #923)
    window_color = QtGui.QApplication.palette().window().color()
    color_table["syswindow"] = (
        window_color.red() / 255.0,
        window_color.green() / 255.0,
        window_color.blue() / 255.0,
        1.0,
    )
