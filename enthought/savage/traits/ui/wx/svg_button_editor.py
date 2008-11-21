
import xml.etree.cElementTree as etree
import wx

from enthought.traits.api import Str, Range, Enum, Instance, Event, Int, Property
from enthought.traits.ui.api import View
from enthought.traits.ui.ui_traits import AView
from enthought.traits.ui.wx.editor import Editor
from enthought.traits.ui.wx.basic_editor_factory import BasicEditorFactory

from enthought.pyface.widget import Widget

from enthought.savage.svg.document import SVGDocument
from enthought.savage.svg.backends.wx.renderer import Renderer
from wx_render_panel import RenderPanel


class ButtonRenderPanel(RenderPanel):
    def __init__(self, parent, button):
        self.button = button
        self.document = button.document
        self.state = 'up'
        
        super(ButtonRenderPanel, self).__init__(parent, document=self.document)
            
    def DoGetBestSize(self):
        return wx.Size(self.button.factory.width, self.button.factory.height)        
        
    def GetBackgroundColour(self):
        bgcolor = super(ButtonRenderPanel, self).GetBackgroundColour()
        if self.state == 'down':
            red, green, blue = bgcolor.Get()
            red -= 50
            green -= 50
            blue -= 50
            bgcolor.Set(red, green, blue, 255)
        return bgcolor
    
    def OnLeftDown(self, evt):
        self.state = 'down'
        self.button.update_editor()
        evt.Skip()
        self.Refresh()

    def OnLeftUp(self, evt):
        self.state = 'up'
        evt.Skip()
        self.Refresh()


#-------------------------------------------------------------------------------
#  '_SVGEditor' class:
#-------------------------------------------------------------------------------
                               
class _SVGButtonEditor ( Editor ):
    """ Traits UI 'display only' image editor.
    """
    
    document = Instance(SVGDocument)
    
    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------
        
    def init ( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        
        tree = etree.parse(self.factory.filename)
        root = tree.getroot()
        
        self.document = SVGDocument(root, renderer=Renderer)
        self.control = ButtonRenderPanel( parent, self)
        
        svg_w, svg_h = self.control.GetBestSize()
        scale_factor = float(svg_w)/self.factory.width
        self.control.zoom /= scale_factor
        self.control.Refresh()
                        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------

    def update_editor ( self ):
        """ Updates the editor when the object trait changes externally to the
            editor.
        """
        factory    = self.factory
        self.value = factory.value
                    
#-------------------------------------------------------------------------------
#  Create the editor factory object:
#-------------------------------------------------------------------------------
class SVGButtonEditor ( BasicEditorFactory ):
    
    # The editor class to be created:
    klass = _SVGButtonEditor
    
    # Value to set when the button is clicked
    value = Property    
    
    label = Str()
    
    filename = Str()
    
    # Extra padding to add to both the left and the right sides
    width_padding = Range( 0, 31, 7 )

    # Extra padding to add to both the top and the bottom sides
    height_padding = Range( 0, 31, 5 )

    # Presentation style
    style = Enum( 'button', 'radio', 'toolbar', 'checkbox' )

    # Orientation of the text relative to the image
    orientation = Enum( 'vertical', 'horizontal' )
    
    # The optional view to display when the button is clicked:
    view = AView
    
    width = Int(32)
    height = Int(32)
    
    #---------------------------------------------------------------------------
    #  Traits view definition:
    #---------------------------------------------------------------------------

    traits_view = View( [ 'value', '|[]' ] )

    #---------------------------------------------------------------------------
    #  Implementation of the 'value' property:
    #---------------------------------------------------------------------------

    def _get_value ( self ):
        return self._value

    def _set_value ( self, value ):
        self._value = value
        if isinstance(value, basestring):
            try:
                self._value = int( value )
            except:
                try:
                    self._value = float( value )
                except:
                    pass

    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, **traits ):
        self._value = 0
        super( SVGButtonEditor, self ).__init__( **traits)
    
    