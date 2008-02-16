"""
In other backends, we expect the UI toolkit to have a main loop running
external to Enable.  This is not the case with Pyglet, which only provides
access to the underlying graphical libraries, and expects the user of the
library to run its own mainloop and manually dispatch events.  Thus, we
provide a simple framework for writing the top-level Application to
interface with the Enable window (or windows).

One can still use the Enable Pyglet backend without using the
EnablePygletApp; the App is very basic and is meant to be easy to subclass
or replace.
"""

from pyglet.window import Window
from pyglet import clock

__all__ = ["get_app", "PygletApp"]

_CurrentApp = None

def get_app():
    """ Returns a reference to the current running Pyglet App, if one exists """
    global _CurrentApp
    return _CurrentApp

class _PygletApp(object):

    def __init__(self, *windows):
        self.windows = windows
        if len(windows) > 0:
            self.main_window = windows[0]
        else:
            self.main_window = None
            self.windows = []

    def set_main_window(self, window):
        """ Sets the main window to use and interact with.  **window**
        should be an instance of enable.pyglet_backend.PygletWindow
        """
        self.main_window = window
        if window not in self.windows:
            self.windows.append(window)
        return

    def add_window(self, window):
        self.windows.append(window)

    def del_window(self, window):
        if window in self.windows:
            self.windows.remove(window)
        return

    def run(self):
        # This is the default implementation from the Pyglet documenation;
        # this is probably not the right way to do this if we want to
        # integrate this window into an app with other windows or an
        # existing mainloop.

        # Initialization/setup
        # TODO: initialize some fonts
        # TODO: initialize the timer

        # Run the mainloop
        exit = False
        while not exit:
            for window in self.windows:
                #window.switch_to()
                window.dispatch_events()
                #window.clear()
                window.draw()
                clock.tick()
                window.flip()
                if window.has_exit:
                    exit = True
        return

def PygletApp(*args, **kw):
    global _CurrentApp
    if _CurrentApp is None:
        _CurrentApp = _PygletApp(*args, **kw)
    return _CurrentApp


