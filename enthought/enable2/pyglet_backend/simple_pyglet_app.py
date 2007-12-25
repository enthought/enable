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



class SimplePygletApp(object):

    def set_main_window(self, window):
        """ Sets the main window to use and interact with.  **window**
        should be an instance of enable.pyglet_backend.PygletWindow
        """
        self.window = window

    def run(self):
        # This is the default implementation from the Pyglet documenation;
        # this is probably not the right way to do this if we want to
        # integrate this window into an app with other windows or an
        # existing mainloop.

        # Initialization/setup
        # TODO: initialize some fonts
        # TODO: initialize the timer

        # Run the mainloop
        while not self.window.has_exit:
            self.window.dispatch_events()
            self.window.clear()
            self.window.draw()
            self.window.flip()
        return



