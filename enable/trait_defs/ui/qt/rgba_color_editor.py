# (C) Copyright 2005-2023 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the various RGBA color editors and the color editor factory, for
    the Qt user interface toolkit.
"""

# -----------------------------------------------------------------------------
#  Imports:
# -----------------------------------------------------------------------------

from pyface.qt import QtGui

try:
    from pyface.ui.qt.color import toolkit_color_to_rgba, rgba_to_toolkit_color
# compatible with pyface < 8.0.0
except ModuleNotFoundError:
    from pyface.ui.qt4.color import toolkit_color_to_rgba, rgba_to_toolkit_color

from traits.trait_base import SequenceTypes

# Note: The ToolkitEditorFactory class imported from color_editor is a
# subclass of the abstract ToolkitEditorFactory class
# (in traitsui.api) with qt-specific methods defined.
# We need to override the implementations of the qt-specific methods here.

try:
    from traitsui.qt.color_editor import (
        ToolkitEditorFactory as BaseColorToolkitEditorFactory,
    )
# compatible with pyface < 8.0.0
except ModuleNotFoundError:
    from traitsui.qt4.color_editor import (
        ToolkitEditorFactory as BaseColorToolkitEditorFactory,
    )

# -----------------------------------------------------------------------------
#  The Qt ToolkitEditorFactory class:
# -----------------------------------------------------------------------------


class ToolkitEditorFactory(BaseColorToolkitEditorFactory):

    def to_qt_color(self, editor):
        """ Gets the PyQt color equivalent of the object trait.
        """
        try:
            color = getattr(editor.object, editor.name + "_")
        except AttributeError:
            color = getattr(editor.object, editor.name)

        if type(color) in SequenceTypes:
            c = rgba_to_toolkit_color(color)
        else:
            c = QtGui.QColor(color)
        return c

    def from_qt_color(self, color):
        """ Gets the application equivalent of a PyQt value.
        """
        return toolkit_color_to_rgba(color)

    def str_color(self, color):
        """ Returns the text representation of a specified color value.
        """
        if type(color) in SequenceTypes:
            return "(%d,%d,%d,%d)" % (
                int(color[0] * 255.0),
                int(color[1] * 255.0),
                int(color[2] * 255.0),
                int(color[3] * 255.0),
            )
        return color


def RGBAColorEditor(*args, **traits):
    return ToolkitEditorFactory(*args, **traits)
