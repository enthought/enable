""" Test app for the Polygon class. """

from enthought.enable import Component, FilledContainer, Scrolled
from enthought.enable.drawing import DragLine
from enthought.enable.wx import Window
from enthought.pyface.api import ApplicationWindow, GUI
from enthought.traits.api import Any

class DragLineTestWindow(ApplicationWindow):
    """ The application window. """

    # Save a reference to our line so we can update its bounds.
    line = Any
    
    ###########################################################################
    # 'Window' interface.
    ###########################################################################

    def _create_contents(self, parent):
        """ Create the contents of the window. """
        
        filled_container = FilledContainer(min_height=1000, min_width=1000)
        self.line = self._create_empty_line()
        filled_container.on_trait_change( self._container_bounds_changed,
                                          'bounds' ) 
        filled_container.add( self.line )
        scrolled = Scrolled(filled_container)
        window = Window(parent, component=scrolled)

        return window.control

    ###########################################################################
    # Private interface.
    ###########################################################################


    def _container_bounds_changed(self, trait, new):
        """ Handle the parent bounds being changed. """

        # Our bounds should be exactly the bounds of the parent so we can
        # handle any events within the parent window.
        self.line.bounds = new
        
        return

    def _create_empty_line(self):
        """ Create an empty DragLine. """
        
        line = DragLine(line_color = (0.0, 0.1, 0.8, 1.0))
        line.on_trait_change(self._line_complete, 'complete')
        line.reset()
        return line
    
    def _line_complete(self, line, event, value):
        """ Handle the completion of the line. """
        
        line.reset()
        
def main():
    # Create the GUI (this does NOT start the GUI event loop).
    gui = GUI()

    # Screen size:
    screen_width = gui.system_metrics.screen_width or 1024
    screen_height = gui.system_metrics.screen_height or 768

    # Create and open the main window.
    window = DragLineTestWindow( title = "Line Test" )
    window.size = ( screen_width / 3, screen_height / 3 )
    window.open()
    
    # Start the GUI event loop.
    gui.event_loop()


###############################################################################
# Program start-up:
###############################################################################

if __name__ == '__main__':
    main()

#### EOF ######################################################################
