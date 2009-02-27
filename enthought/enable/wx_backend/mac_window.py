""" Defines the concrete top-level Enable 'Window' class for the wxPython GUI
toolkit, based on the kiva driver for OS X.
"""

# Major library imports.
import wx

# Enthought library imports.
from enthought.kiva.backend_wx import gc_for_dc

# Local imports.
from window import Window


class MacWindow(Window):
    """ An Enable Window for wxPython GUIs on OS X.
    """

    #### 'Window' interface ####################################################

    def __init__(self, parent, wid=-1, pos=wx.DefaultPosition,
        size=wx.DefaultSize, **traits):

        Window.__init__(self, parent, wid=wid, pos=pos, size=size, **traits)

        #self.memDC = wx.MemoryDC()
        return

    def _create_gc(self, size, pix_format="bgra32"):
        self.dc = wx.ClientDC(self.control)
        gc = gc_for_dc(self.dc)
        gc.begin()
        return gc

    def _window_paint(self, event):
        self.dc = None
        self._gc = None  # force a new gc to be created for the next paint()


    #### 'AbstractWindow' interface ############################################
 
    def _paint(self, event=None):
        size = self._get_control_size()
        if (self._size != tuple(size)) or (self._gc is None):
            self._gc = self._create_gc(size)
            self._size = tuple(size)
        gc = self._gc
        gc.begin()
        gc.clear(self.bg_color_)
        if hasattr(self.component, "do_layout"):
            self.component.do_layout()
        self.component.draw(gc, view_bounds=(0, 0, size[0], size[1]))
        self._window_paint(event)
        gc.end()
        return
 


#### EOF #######################################################################
