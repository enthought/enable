
import wx

from enthought.traits.api import Event, Float, Instance, Int
from enthought.traits.ui.api import Item, View

from enthought.enable import Component, Container
from enthought.enable.wx import Window
from enthought.enable.enable_traits import red_color_trait

class SimpleComponent(Component):

    color = red_color_trait
    myview = View(Item("color"))
    
    def _draw(self, gc):
        gc.save_state()
        gc.set_fill_color(self.color_)
        gc.rect(*self.bounds)
        gc.fill_path()
        gc.restore_state()

class Working(SimpleComponent):

    def normal_left_up(self, event):
        print "Left up!"


class Broken(SimpleComponent):
    
    def normal_left_up(self, event):
        self.edit_traits(kind="modal")
        self.redraw()


class TestContainer(Container):
    def _draw_container(self, gc):
        gc.save_state()
        gc.set_fill_color((0.5, 0.5, 0.5, 1.0))  # gray
        gc.rect(*self.bounds)
        gc.fill_path()
        gc.restore_state()


class EnableWindowFrame ( wx.Frame ):
    def __init__ ( self, component, *args, **kw ):
        wx.Frame.__init__( *(self,) + args, **kw )
        sizer = wx.BoxSizer( wx.HORIZONTAL )
        self.enable_window = Window( self, -1, component = component )
        sizer.Add( self.enable_window.control, 1, wx.EXPAND )
        self.SetSizer( sizer )
        self.SetAutoLayout( True )
        self.Show( True )


if __name__ == "__main__":
    app = wx.PySimpleApp()
    
    box = Working(bounds=(50,50,100,100))
    
    #box = Broken(bounds=(50,50,100,100))
    
    mycontainer = TestContainer(box, bounds=(0,0,200,200))
    frame = EnableWindowFrame(mycontainer, None, -1,
                            "Click handler demo", size=wx.Size(300,300))
    app.SetTopWindow(frame)
    frame.Show(True)
    app.MainLoop()
