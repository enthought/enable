""" Defines a Traits editor for displaying an Enable component.
"""
#-------------------------------------------------------------------------------
#  Written by: David C. Morrill
#  Date: 01/26/2007
#  (c) Copyright 2007 by Enthought, Inc.
#----------------------------------------------------------------------------

from enthought.enable.colors import ColorTrait
from enthought.enable.window import Window

from traits.etsconfig.api import ETSConfig

from enthought.traits.api import Property, Tuple
from enthought.traits.ui.api import BasicEditorFactory

if ETSConfig.toolkit == 'wx':
    from enthought.traits.ui.wx.editor import Editor
elif ETSConfig.toolkit == 'qt4':
    from enthought.traits.ui.qt4.editor import Editor
else:
    Editor = object

class _ComponentEditor( Editor ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # The plot editor is scrollable (overrides Traits UI Editor).
    scrollable = True

    def init( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
        widget.
        """
        self._window = Window( parent, size=self.factory.size, component=self.value )
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

    def update_editor( self ):
        """ Updates the editor when the object trait changes externally to the
        editor.
        """
        self._window.component = self.value
        return


class ComponentEditor( BasicEditorFactory ):
    """ wxPython editor factory for Enable components.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # The class used to create all editor styles (overrides BasicEditorFactory).
    klass = _ComponentEditor

    # The background color for the window
    bgcolor = ColorTrait('sys_window')

    # The default size of the Window wrapping this Enable component
    size = Tuple((400,400))

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
