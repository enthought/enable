""" Defines the concrete top-level Enable 'Window' class for the wxPython GUI
toolkit, based on the kiva driver for OS X.
"""

# Major library imports.
import wx

# Enthought library imports.
from enthought.kiva.mac import ABCGI, get_macport

# Local imports.
from window import Window


class MacWindow(Window):
    """ An Enable Window for wxPython GUIs on OS X.
    """

    #### 'Window' interface ####################################################

    def __init__(self, parent, wid=-1, pos=wx.DefaultPosition,
        size=wx.DefaultSize, **traits):

        Window.__init__(self, parent, wid=wid, pos=pos, size=size, **traits)

        self.memDC = wx.MemoryDC()
        return


    def _create_gc(self, size, pix_format="bgra32"):
        """ Create a Kiva graphics context of a specified size.

        The pix_format argument is for interface compatibility and will be
        ignored.
        """

        if self._gc is not None:
            self._gc.end()

        self.bitmap = wx.EmptyBitmap(size[0], size[1])
        self.memDC.SelectObject(self.bitmap)
        port = get_macport(self.memDC)
        gc = ABCGI.CGContextForPort(port)
        gc.begin()
        return gc
 

    def _window_paint(self, event):
        """ Do a GUI toolkit specific screen update.
        """
        # NOTE: This should do an optimal update based on the value of the 
        # self._update_region, but Kiva does not currently support this:
        control = self.control
        wdc = control._dc = wx.PaintDC(control)
        wdc.Blit(0, 0, self._size[0], self._size[1], self.memDC, 0, 0)

        control._dc = None
        return


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
