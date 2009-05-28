"""
Support class that wraps up the boilerplate toolkit calls that virtually all
demo programs have to use.
"""

from enthought.etsconfig.api import ETSConfig

# FIXME - it should be enough to do the following import, but because of the
# PyQt/traits problem (see below) we can't because it would drag in traits too
# early.  Until it is fixed we just assume wx if we can import it.
# Force the selection of a valid toolkit.
#import enthought.enable.toolkit 
if not ETSConfig.enable_toolkit:
    for toolkit, toolkit_module in (('wx', 'wx'), ('qt4', 'PyQt4')):
        try:
            exec "import " + toolkit_module
            ETSConfig.enable_toolkit = toolkit
            break
        except ImportError:
            pass
    else:
        raise RuntimeError("Can't load wx or qt4 backend for Chaco.")


if ETSConfig.enable_toolkit == 'wx':
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
            "Subclasses should override this method and return an enable.Window"
            raise NotImplementedError


    def demo_main(demo_class, size=(400,400), title="Enable Demo"):
        "Takes the class of the demo to run as an argument."
        app = wx.GetApp()
        if app is None:
            app = wx.PySimpleApp()
        frame = demo_class(None, size=size, title=title)
        app.SetTopWindow(frame)
        app.MainLoop()

elif ETSConfig.enable_toolkit == 'qt4':
    from PyQt4 import QtGui

    _app = QtGui.QApplication.instance()

    if _app is None:
        import sys
        _app = QtGui.QApplication(sys.argv)

    class DemoFrame(QtGui.QWidget):
        def __init__ (self, parent, **kw):
            QtGui.QWidget.__init__(self)

            # Create the subclass's window
            self.enable_win = self._create_window()

            layout = QtGui.QVBoxLayout()
            layout.setMargin(0)
            layout.addWidget(self.enable_win.control)

            self.setLayout(layout)

            if 'size' in kw:
                self.resize(*kw['size'])

            if 'title' in kw:
                self.setWindowTitle(kw['title'])

            self.show()

        def _create_window(self):
            "Subclasses should override this method and return an enable.Window"
            raise NotImplementedError


    def demo_main(demo_class, size=(400,400), title="Enable Demo"):
        "Takes the class of the demo to run as an argument."
        frame = demo_class(None, size=size, title=title)
        _app.exec_()


elif ETSConfig.enable_toolkit == 'pyglet':

    from pyglet import app
    from pyglet import clock

    class DemoFrame(object):
        def __init__(self):
            if app:
                window = self._create_window()
                if window:
                    self.enable_win = window
                else:
                    self.enable_win = None
            return

        def _create_window(self):
            raise NotImplementedError

    def demo_main(demo_class, size=(640,480), title="Enable Example"):
        """ Runs a simple application in Pyglet using an instance of
        **demo_class** as the main window or frame.

        **demo_class** should be a subclass of DemoFrame or the pyglet
        backend's Window class.
        """
        if issubclass(demo_class, DemoFrame):
            frame = demo_class()
            if frame.enable_win is not None:
                window = frame.enable_win.control
            else:
                window = None
        else:
            window = demo_class().control

        if window is not None:
            if not window.fullscreen:
                window.set_size(*size)
            window.set_caption(title)

        app.run()
        


# EOF
