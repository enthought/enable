# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Traits UI button editor for SVG images.
"""

# -----------------------------------------------------------------------------
#  Imports:
# -----------------------------------------------------------------------------
import os.path

from traits.api import Bool, Any, Str
from traitsui.qt4.editor import Editor

from pyface.qt import QtCore, QtGui

# add the Qt's installed dir plugins to the library path so the iconengines
# plugin will be found:
qt_plugins_dir = os.path.join(os.path.dirname(QtCore.__file__), "plugins")
QtCore.QCoreApplication.addLibraryPath(qt_plugins_dir)

# -----------------------------------------------------------------------------
#  'SVGButtonEditor' class:
# -----------------------------------------------------------------------------


class SVGButtonEditor(Editor):

    icon = Any
    toggled_icon = Any
    toggle_label = Str
    toggle_tooltip = Str
    toggle_state = Bool

    # -------------------------------------------------------------------------
    #  Editor interface
    # -------------------------------------------------------------------------

    def init(self, parent):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        self.icon = QtGui.QIcon(self.factory.filename)
        if self.factory.toggle_filename:
            self.toggled_icon = QtGui.QIcon(self.factory.toggle_filename)

        if self.factory.toggle_label != "":
            self.toggle_label = self.factory.toggle_label
        else:
            self.toggle_label = self.factory.label

        if self.factory.toggle_tooltip != "":
            self.toggle_tooltip = self.factory.toggle_tooltip
        else:
            self.toggle_tooltip = self.factory.tooltip

        control = self.control = QtGui.QToolButton()
        control.setAutoRaise(True)
        control.setIcon(self.icon)
        control.setText(self.factory.label)
        control.setIconSize(
            QtCore.QSize(self.factory.width, self.factory.height)
        )

        if self.factory.label:
            if self.factory.orientation == "horizontal":
                control.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            else:
                control.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        else:
            control.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        if self.factory.toggle:
            control.setCheckable(True)
            control.toggled.connect(self._toggle_button)

        control.clicked.connect(self.update_object)

        if self.factory.tooltip:
            control.setToolTip(self.factory.tooltip)
        else:
            self.set_tooltip()

    def prepare(self, parent):
        """ Finishes setting up the editor. This differs from the base class
            in that self.update_editor() is not called at the end, which
            would fire an event.
        """
        name = self.extended_name
        if name != "None":
            self.context_object.on_trait_change(
                self._update_editor, name, dispatch="ui"
            )
        self.init(parent)
        self._sync_values()

    def update_object(self):
        """ Handles the user clicking the button by setting the factory value
            on the object.
        """
        self.value = self.factory.value

    def update_editor(self):
        """ Updates the editor when the object trait changes externally to the
            editor.
        """
        pass

    def _toggle_button(self):
        self.toggle_state = not self.toggle_state
        if self.toggle_state and self.toggled_icon:
            self.control.setIcon(self.toggled_icon)
            self.control.setText(self.toggle_label)
            self.control.setToolTip(self.toggle_tooltip)

        elif not self.toggle_state and self.toggled_icon:
            self.control.setIcon(self.icon)
            self.control.setText(self.factory.label)
            self.control.setToolTip(self.factory.tooltip)
