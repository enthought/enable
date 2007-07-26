"""
Support class that wraps up the boilerplate wx calls that virtually all
demo programs have to use.
"""

import wx

class DemoFrame(wx.Frame):
    def __init__ ( self, *args, **kw ):
        wx.Frame.__init__( *(self,) + args, **kw )
        #self.SetTitle("Enable Demo")
        self.SetAutoLayout( True )
    
        # Create the subclass's window
        self.enable_win = self._create_window()
        
        # Listen for the Activate event so we can restore keyboard focus.
        wx.EVT_ACTIVATE( self, self._on_activate )
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.enable_win.control, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Show( True )
        return

    def _on_activate(self, event):
        if self.enable_win is not None and self.enable_win.control is not None:
            self.enable_win.control.SetFocus()

    def _create_window(self):
        "Subclasses should override this method and return an enable.wx.Window"
        raise NotImplementedError


def demo_main(demo_class, size=(400,400), title="Enable Demo"):
    "Takes the class of the demo to run as an argument."
    app = wx.PySimpleApp()
    frame = demo_class(None, size=size, title=title)
    app.SetTopWindow(frame)
    app.MainLoop()

# EOF
