import wx

from enthought.kiva import Canvas

class MyCanvas(Canvas):
    
    def __init__(self, parent, id=-1, size=wx.DefaultSize):
        Canvas.__init__(self, parent, id, size=size)
        return
    
    def do_draw(self, gc):
        w = gc.width()
        h = gc.height()
        # Draw a red box with green border
        gc.rect( w/4, h/4, w/2, h/2 )
        gc.set_line_width(5.0)
        gc.set_stroke_color( (0.0, 1.0, 0.0, 1.0) )
        gc.set_fill_color( (1.0, 0.0, 0.0, 1.0) )
        gc.draw_path()
        return

class MyWindow(wx.Frame):
    def __init__(self, id=-1, title="Simple Kiva.wx example", size=(500,500)):
        parent = None
        wx.Frame.__init__(self, parent, id, title, size=size)
        canvas = MyCanvas(self)
        self.Show(1)
        return

if __name__ == "__main__":
    class MyApp(wx.App):
        def OnInit(self):
            MyWindow(size=(500,500))
            return 1
    
    app = MyApp()
    app.MainLoop()
