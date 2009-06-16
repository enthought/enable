import wx
import numpy

from enthought.kiva import Canvas

class MyCanvas(Canvas):

    def __init__(self, parent, id=-1, size=wx.DefaultSize):
        Canvas.__init__(self, parent, id, size=size)
        return

    def do_draw(self, gc):
        # colors are 5 doubles: offset, red, green, blue, alpha
        starting_color = numpy.array([0.0, 1.0, 1.0, 1.0, 1.0])
        ending_color = numpy.array([0.0, 0.0, 0.0, 0.0, 1.0])

        gc.clear()
        gc.rect(100,100,300,300)
        gc.linear_gradient(100, 100, 300, 300,
                            numpy.array([starting_color, ending_color]),
                            2, "")
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
