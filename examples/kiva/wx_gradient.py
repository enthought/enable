from __future__ import with_statement

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
        ending_color = numpy.array([1.0, 0.0, 0.0, 0.0, 1.0])

        gc.clear()

        # diagonal
        with gc:
            gc.rect(50,25,150,100)
            gc.linear_gradient(50,25,150,125,
                                numpy.array([starting_color, ending_color]),
                                "pad", 'objectBoundingBox')
            gc.draw_path()

        # vertical
        with gc:
            gc.rect(50,150,150,100)
            gc.linear_gradient(50,150,50,250,
                                numpy.array([starting_color, ending_color]),
                                "pad", 'objectBoundingBox')
            gc.draw_path()

        # horizontal
        with gc:
            gc.rect(50,275,150,100)
            gc.linear_gradient(50,275,150,275,
                                numpy.array([starting_color, ending_color]),
                                "pad", 'objectBoundingBox')
            gc.draw_path()
        
        # radial
        with gc:
            gc.arc(325, 75, 50, 0.0, 2*numpy.pi)
            gc.radial_gradient(325, 75, 50, 325, 75,
                                numpy.array([starting_color, ending_color]),
                                "pad", 'objectBoundingBox')
            gc.draw_path()

        # radial with focal point in upper left
        with gc:
            gc.arc(325, 200, 50, 0.0, 2*numpy.pi)
            gc.radial_gradient(325, 200, 50, 300, 225,
                            numpy.array([starting_color, ending_color]),
                            "pad", 'objectBoundingBox')
            gc.draw_path()

        # radial with focal point in bottom right
        with gc:
            gc.arc(325, 325, 50, 0.0, 2*numpy.pi)
            gc.radial_gradient(325, 325, 50, 350, 300,
                                numpy.array([starting_color, ending_color]),
                                "pad", 'objectBoundingBox')
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
            MyWindow(size=(500,400))
            return 1

    app = MyApp()
    app.MainLoop()
