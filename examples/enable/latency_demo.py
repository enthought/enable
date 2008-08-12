"""
Test to see what level of click latency is noticeable.
"""

import time
import wx

from enthought.traits.api import Float

from enthought.enable.api import Component, Container, ColorTrait, black_color_trait
from enthought.enable.wx_backend.api import Window

from enthought.kiva import Font

try:
    font = Font(face_name="Arial")
except:
    font = Font(face_name="SWISS")

class Box(Component):
    color = ColorTrait("red")

    delay = Float(0.50)
    
    def _draw_mainlayer(self, gc, view=None, mode="default"):
        if self.event_state == "clicked":
            print "waiting %0.4f seconds... " % self.delay,
            time.sleep(self.delay)
            print "done."
            
            gc.save_state()
            gc.set_fill_color(self.color_)
            gc.rect(*(self.position + self.bounds))
            gc.fill_path()
            gc.restore_state()
            
        else:
            gc.save_state()
            gc.set_stroke_color(self.color_)
            gc.set_fill_color(self.color_)
            gc.set_line_width(1.0)
            gc.rect(*(self.position + self.bounds))
            gc.stroke_path()
            
            gc.set_font(font)
            x,y = self.position
            dx,dy = self.bounds
            tx, ty, tdx, tdy = gc.get_text_extent(str(self.delay))
            gc.set_text_position(x+dx/2-tdx/2, y+dy/2-tdy/2)
            gc.show_text(str(self.delay))
            gc.restore_state()
    
    def normal_left_down(self, event):
        self.event_state = "clicked"
        event.handled = True
        self.request_redraw()
        
    def clicked_left_up(self, event):
        self.event_state = "normal"
        event.handled = True
        self.request_redraw()

class MyContainer(Container):
    text_color = black_color_trait
    
    def _draw_container_mainlayer(self, gc, view_bounds=None, mode="default"):
        s = "Hold down the mouse button on the boxes."
        gc.save_state()
        gc.set_font(font)
        gc.set_fill_color(self.text_color_)
        tx, ty, tdx, tdy = gc.get_text_extent(s)
        x,y = self.position
        dx,dy = self.bounds
        gc.set_text_position(x+dx/2-tdx/2, y+dy-tdy-20)
        gc.show_text(s)
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
    times_and_bounds = { 0.5 : (60,200,100,100),
                            0.33 : (240,200,100,100),
                            0.25: (60,50,100,100),
                            0.10: (240,50,100,100) }
    
    container = MyContainer(auto_size = False)
    for delay, bounds in times_and_bounds.items():
        box = Box()
        container.add(box)
        box.position = list(bounds[:2])
        box.bounds = list(bounds[2:])
        box.delay = delay
    frame = EnableWindowFrame(container, None, -1,
                            "Latency Test - Click a box", size=wx.Size(400,400))
    app.SetTopWindow(frame)
    frame.Show(True)
    app.MainLoop()

