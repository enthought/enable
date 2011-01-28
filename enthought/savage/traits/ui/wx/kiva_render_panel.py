import wx

from enthought.savage.svg.backends.kiva import renderer
from enthought.savage.svg.document import SVGDocument
from enthought.enable.api import Container, Window
from enthought.traits.api import Instance, Float

class KivaContainer(Container):

    document = Instance(SVGDocument)
    zoom = Float(100.0)

    def draw(self, gc, view_bounds=None, mode="default"):
        gc.clear()
        if not self.document:
            gc.show_text_at_point("No Document", 20, 20)
            return

        with gc:
            # SVG origin is upper right with y positive is down. argh.
            # Set up the transforms to fix this up.
            gc.translate_ctm(0, gc.height())
            # zoom percentage
            scale = float(self.zoom) / 100.0
            gc.scale_ctm(scale, -scale)
            self.document.render(gc)



class RenderPanel(wx.Window):
    def __init__(self, parent, document=None):
        super(RenderPanel, self).__init__(parent)

        self.document = document
        if self.document is not None:
            self.document.renderer = renderer

        self.container = KivaContainer(document=self.document)

        size = wx.Size(200,200)
        if document is not None:
            size = document.getSize()

        self._window = Window( parent=self, size=size, component=self.container )
        self.control = self._window.control
        self._parent = parent

        wx.EVT_PAINT(self, self.OnPaint)

    def OnPaint(self, event):
        # set the bg color of the window to match the container, otherwise
        # we'll get a border. This might not always be what we want though
        self.SetBackgroundColour([int(255*c) for c in self.container.bgcolor_])

        self.container.draw(self._window._gc)

    def GetBestSize(self):
        if not self.document:
            return (-1,-1)

        return self.document.getSize()
