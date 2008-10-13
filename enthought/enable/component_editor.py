""" Defines a Traits editor for displaying an Enable component.
"""
#-------------------------------------------------------------------------------
#  Written by: David C. Morrill
#  Date: 01/26/2007
#  (c) Copyright 2007 by Enthought, Inc.
#----------------------------------------------------------------------------

from enthought.enable.colors import ColorTrait

from enthought.etsconfig.api import ETSConfig

from enthought.traits.api import Property, Tuple
from enthought.traits.ui.api import BasicEditorFactory

if ETSConfig.toolkit == 'wx':
    from enthought.traits.ui.wx.editor import Editor
    from enthought.enable.wx_backend.api import Window
elif ETSConfig.toolkit == 'qt4':
    from enthought.traits.ui.qt4.editor import Editor
    from enthought.enable.qt4_backend.api import Window
else:
    Editor = object
    Window = None

class _ComponentEditor( Editor ):

    #---------------------------------------------------------------------------
    #  Trait definitions:     
    #---------------------------------------------------------------------------

    # The plot editor is scrollable (overrides Traits UI Editor).
    scrollable = True

    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------
    def init( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
        widget.
        """
        self._window = Window( parent, size=self.factory.size, component=self.value )
        self.control = self._window.control
        self._window.bg_color = self.factory.bgcolor
        self._parent = parent

    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes externally to the editor:
    #---------------------------------------------------------------------------
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
