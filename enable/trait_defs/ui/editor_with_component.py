# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

from traits.api import Instance, observe
from traitsui.api import toolkit_object

from enable.component import Component
from enable.enable_traits import font_trait
from enable.label import Label
from enable.window import Window


#: The toolkit's Editor base class
Editor = toolkit_object("editor:Editor")


class EditorWithComponent(Editor):
    """Base class for editors which hold a Component that displays the value.

    This is distinct from CompononentEditor in that the value is not the
    Component.
    """

    #: The window that displays the component.
    window = Instance(Window)

    #: The component that is created by the UI.
    component = Instance(Component)

    def init(self, parent):
        self.component = self.create_component()
        size = self._get_initial_size()
        self.window = self.create_window(parent, size)
        self.control = self.window.control
        self._parent = parent

    def create_component(self):
        raise NotImplementedError()

    def create_window(self, parent, size):
        window = Window(
            parent,
            size=size,
            component=self.component,
            high_resolution=self.factory.high_resolution,
            bgcolor='sys_window',
        )
        return window

    def dispose(self):
        if self.window is not None:
            self.window.cleanup()
            self.component = None
            self.window = None
            self._parent = None
        super().dispose()


class EditorWithLabelComponent(EditorWithComponent):
    """A class that creates a Label component.

    By default it displays the string representation of the value.
    """

    #: The font to use for the label.
    font = font_trait()

    def create_component(self):
        """Creates the label component."""
        component = Label(
            hjustify="center",
            vjustify="center",
            resizable='hv',
            text=self.str_value,
            font=self.font,
        )
        return component

    def update_editor(self):
        if self.component is not None:
            self.component.text = self.str_value
            self.component.invalidate_and_redraw()

    @observe('font')
    def update_font(self, event):
        if self.component is not None:
            self.component.font = self.font
            self.component.invalidate_and_redraw()

    def _get_initial_size(self):
        width = self.item.width
        height = self.item.height

        if width < 0:
            width = 200
        if height < 0:
            height = 50

        return width, height

    def set_size_policy(self, direction, resizable, springy, stretch):
        """Set the size policy of the editor's component.

        This is only used by the Qt backend.  This is always springy and
        resizable.
        """
        from pyface.qt import QtGui

        policy = self.window.control.sizePolicy()

        if direction == QtGui.QBoxLayout.Direction.LeftToRight:
            policy.setHorizontalStretch(stretch)
            policy.setHorizontalPolicy(QtGui.QSizePolicy.Policy.Expanding)
            policy.setVerticalStretch(stretch)
            policy.setVerticalPolicy(QtGui.QSizePolicy.Policy.Expanding)

        else:  # TopToBottom
            policy.setVerticalStretch(stretch)
            policy.setVerticalPolicy(QtGui.QSizePolicy.Policy.Expanding)
            policy.setHorizontalStretch(stretch)
            policy.setHorizontalPolicy(QtGui.QSizePolicy.Policy.Expanding)

        self.window.control.setSizePolicy(policy)
        if self.window.control is not self.control:
            super().set_size_policy()
