# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines a Traits editor for displaying an Enable component.
"""

from enable.colors import ColorTrait
from enable.window import Window

from traits.etsconfig.api import ETSConfig

from traits.api import Bool, Property, Tuple
from traitsui.api import BasicEditorFactory

if ETSConfig.toolkit == "wx":
    from traitsui.wx.editor import Editor
elif ETSConfig.toolkit.startswith("qt"):
    from traitsui.qt4.editor import Editor
else:
    Editor = object


class _ComponentEditor(Editor):

    # -------------------------------------------------------------------------
    #  Trait definitions:
    # -------------------------------------------------------------------------

    # The plot editor is scrollable (overrides Traits UI Editor).
    scrollable = True

    def init(self, parent):
        """ Finishes initializing the editor by creating the underlying toolkit
        widget.
        """

        size = self._get_initial_size()

        self._window = Window(
            parent,
            size=size,
            component=self.value,
            high_resolution=self.factory.high_resolution,
        )

        self.control = self._window.control
        self._window.bgcolor = self.factory.bgcolor
        self._parent = parent

    def dispose(self):
        """ Disposes of the contents of an editor.
        """
        self._window.cleanup()
        self._window = None
        self._parent = None
        super(_ComponentEditor, self).dispose()

    def update_editor(self):
        """ Updates the editor when the object trait changes externally to the
        editor.
        """
        self._window.component = self.value

    def _get_initial_size(self):
        """ Compute the initial size of the component.

        Use the item size to set the size of the component;
        if not specified, use the default size given in ComponentEditor.size
        """

        width = self.item.width
        if width < 0:
            width = self.factory.size[0]

        height = self.item.height
        if height < 0:
            height = self.factory.size[1]

        return width, height


class ComponentEditor(BasicEditorFactory):
    """ wxPython editor factory for Enable components.
    """

    # -------------------------------------------------------------------------
    #  Trait definitions:
    # -------------------------------------------------------------------------

    # The class used to create all editor styles (overrides BasicEditorFactory)
    klass = _ComponentEditor

    # The background color for the window
    bgcolor = ColorTrait("sys_window")

    # When available, use HiDPI for GraphicsContext rasterization.
    high_resolution = Bool(True)

    # The default size of the Window wrapping this Enable component
    size = Tuple((400, 400))

    # Convenience function for accessing the width
    width = Property

    # Convenience function for accessing the width
    height = Property

    def _get_width(self):
        return self.size[0]

    def _set_width(self, width):
        self.size = (width, self.size[1])

    def _get_height(self):
        return self.size[1]

    def _set_height(self, height):
        self.size = (self.size[0], height)
