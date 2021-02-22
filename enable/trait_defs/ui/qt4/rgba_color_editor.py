# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the various RGBA color editors and the color editor factory, for
    the Qt4 user interface toolkit.
"""

# -----------------------------------------------------------------------------
#  Imports:
# -----------------------------------------------------------------------------

from pyface.qt import QtGui

from traits.trait_base import SequenceTypes

# Note: The ToolkitEditorFactory class imported from color_editor is a
# subclass of the abstract ToolkitEditorFactory class
# (in traitsui.api) with qt4-specific methods defined.
# We need to override the implementations of the qt4-specific methods here.
from traitsui.qt4.color_editor import (
    ToolkitEditorFactory as BaseColorToolkitEditorFactory,
)

# -----------------------------------------------------------------------------
#  The PyQt4 ToolkitEditorFactory class:
# -----------------------------------------------------------------------------


class ToolkitEditorFactory(BaseColorToolkitEditorFactory):
    def to_qt4_color(self, editor):
        """ Gets the PyQt color equivalent of the object trait.
        """
        try:
            color = getattr(editor.object, editor.name + "_")
        except AttributeError:
            color = getattr(editor.object, editor.name)

        if type(color) in SequenceTypes:
            c = QtGui.QColor()
            c.setRgbF(color[0], color[1], color[2], color[3])
        else:
            c = QtGui.QColor(color)
        return c

    def from_qt4_color(self, color):
        """ Gets the application equivalent of a PyQt value.
        """
        return (color.redF(), color.greenF(), color.blueF(), color.alphaF())

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
