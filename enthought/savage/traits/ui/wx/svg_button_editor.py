
import xml.etree.cElementTree as etree
import wx

from enthought.traits.api import Str, Range, Enum, Instance, Event, Int
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
        
        super(ButtonRenderPanel, self).__init__(parent, document=self.document)
    
    def OnLeftDown(self, evt):
        print "left down!"
        evt.Skip()

    def OnLeftUp(self, evt):
        print "left up!"
        evt.Skip()


class _SVGButtonControl(Widget):
    
    document = Instance(SVGDocument)
    
    # The (optional) label:
    label = Str
    
    # Extra padding to add to both the left and right sides:
    width_padding = Range( 0, 31, 7 )
    
    # Extra padding to add to both the top and bottom sides:
    height_padding = Range( 0, 31, 5 )
    
    # Presentation style:
    style = Enum( 'button', 'radio', 'toolbar', 'checkbox' )
    
    # Orientation of the text relative to the image:
    orientation = Enum( 'vertical', 'horizontal' )
    
    # Fired when a 'button' or 'toolbar' style control is clicked: 
    clicked = Event
    
    width = Int(32)
    height = Int(32)
    
    def __init__ ( self, parent, **traits ):
        """ Creates a new image control. 
        """
        super( _SVGButtonControl, self ).__init__( **traits )
                                  
        panel = wx.Window( parent, -1, (self.width, self.height))
                                  
        border = self.width_padding
        border_flag = wx.RIGHT
        if self.orientation == 'vertical':
            sizer = wx.BoxSizer(wx.VERTICAL)
            border = self.height_padding
            border_flag = wx.BOTTOM
        else:
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            
        panel.SetSizer(sizer)
        panel.SetAutoLayout(True)
        panel.SetBackgroundColour(wx.NullColor)
        
        render_panel = ButtonRenderPanel(parent, self)
        
        if self.label != '':
            label_control = wx.StaticText(panel, -1, self.label)
            sizer.Add(label_control, 0, border_flag, 0)
            
            label_size = label_control.GetSize()
            if self.orientation == 'vertical':
                width = max(self.width, label_size.x + 2)
                height = self.height + label_size.y + 2
            else:
                height = max(self.height, label_size.y + 2)
                width = self.width + label_size.x + 2
            
            panel.SetSize((width, height))
                                  
        sizer.Add(render_panel, 0, border_flag, border)
                                  
        self.control = panel
        self.control._owner = self
        self._mouse_over    = self._button_down = False
 
        # Set up mouse event handlers:
        #wx.EVT_ENTER_WINDOW( self.control, self._on_enter_window )
        #wx.EVT_LEAVE_WINDOW( self.control, self._on_leave_window )
        #wx.EVT_LEFT_DOWN(    self.control, self._on_left_down )
        #wx.EVT_LEFT_UP(      self.control, self._on_left_up )
        #wx.EVT_PAINT(        self.control, self._on_paint )

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
        #self._control = _SVGButtonControl( parent, document=document, label=self.factory.label)
        #self.control = self._control.control
        self.control = ButtonRenderPanel( parent, self)
        
        svg_w, svg_h = self.control.GetBestSize()
        scale_factor = svg_w/self.factory.width
        self.control.zoom /= scale_factor
        self.control.Refresh()
                        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------

    def update_editor ( self ):
        """ Updates the editor when the object trait changes externally to the
            editor.
        """
        pass
        #self.control.Refresh()
                    
#-------------------------------------------------------------------------------
#  Create the editor factory object:
#-------------------------------------------------------------------------------
class SVGButtonEditor ( BasicEditorFactory ):
    
    # The editor class to be created:
    klass = _SVGButtonEditor
    
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
        super( SVGButtonEditor, self ).__init__( **traits )
    
    