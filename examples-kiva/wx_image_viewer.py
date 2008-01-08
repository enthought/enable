import wx
import sys
import os
from enthought.kiva import Canvas
from enthought.kiva.agg import GraphicsContextArray
from enthought.kiva.backend_image import Image

#Event ID numbers
ID_OPEN = 101
ID_QUIT = 102

class ImageCanvas(Canvas):
    def __init__(self, parent, id=-1, size=wx.DefaultSize):
        Canvas.__init__(self, parent, id, size=size)

        # Until we have an image, use a blank GraphicsContextArray
        self.img = GraphicsContextArray((800, 600))
        return

    def load_image(self, filename):
        # Create a new Image object for the new file and then make wx repaint the window
        self.img = Image(filename)
        self.dirty = 1
        self.Refresh()
    
    def do_draw(self, gc):
        # Use Image's abillity to draw itself onto a gc to paint the window
        gc.draw_image(self.img, (0,0,self.img.width(), self.img.height()))

class ImageViewerWindow(wx.Frame):
    def __init__(self, id=-1, title="Image Viewer", size=(800,600)):
        parent = None
        wx.Frame.__init__(self, parent, id, title, size=size)

        # Setup the menubar and file menu
        filemenu = wx.Menu()
        filemenu.Append(ID_OPEN, "&Open", "Open an image file to view")
        filemenu.Append(ID_QUIT, "&Quit", "Quit the program")
        menubar = wx.MenuBar()
        menubar.Append(filemenu, "&File")
        self.SetMenuBar(menubar)

        # Register the event IDs for each menu item to call the appropriate function
        wx.EVT_MENU(self, ID_QUIT, self.onQuit)
        wx.EVT_MENU(self, ID_OPEN, self.onOpen)
        
        #Create an ImageCanvas to hold the image
        self.canvas = ImageCanvas(self)

        self.Show(1)
        return
    

    def onQuit(self, e):
        sys.exit()
        
    def onOpen(self, e):
        #Create a file dialog
        dlg = wx.FileDialog(
            self, message = "Choose an image", defaultDir = os.getcwd(),
            defaultFile = "", wildcard = "All Files (*.*)|*.*",
            style=wx.OPEN | wx.CHANGE_DIR
            )
        #Show it modally.  If the user clicks OK, have the ImageCanvas load the
        # new file.
        if dlg.ShowModal() ==wx.ID_OK:
            paths = dlg.GetPaths()
            self.canvas.load_image(paths[0])
        

if __name__ == "__main__":
    class MyApp(wx.App):
        def OnInit(self):
            ImageViewerWindow(size=(800,600))
            return 1
    
    app = MyApp(0)
    app.MainLoop()
