""" Test app for the Polygon class. """

from enthought.enable import Component, FilledContainer, Scrolled
from enthought.enable.drawing import PointPolygon
from enthought.enable.wx import Window
from enthought.pyface.api import ApplicationWindow, GUI
from enthought.traits.api import Any

class PointPolygonTestWindow(ApplicationWindow):
    """ The application window. """

    # Save a reference to our polygon so we can update its bounds.
    polygon = Any
    
    ###########################################################################
    # 'Window' interface.
    ###########################################################################

    def _create_contents(self, parent):
        """ Create the contents of the window. """
        
        filled_container = FilledContainer(min_height=1000, min_width=1000)
        self.polygon = self._create_empty_polygon()
        filled_container.on_trait_change( self._container_bounds_changed,
                                          'bounds' ) 
        filled_container.add( self.polygon )
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
        self.polygon.bounds = new
        
        return
    
    def _create_empty_polygon(self):
        """ Create an empty PointPolygon. """
        
        polygon = PointPolygon(background_color = (0.0, 0.4, 1.0, 1.0),
                               border_color = (0.0, 0.0, 0.2, 1.0))

        return polygon
    
def main():
    # Create the GUI (this does NOT start the GUI event loop).
    gui = GUI()

    # Screen size:
    screen_width = gui.system_metrics.screen_width or 1024
    screen_height = gui.system_metrics.screen_height or 768

    # Create and open the main window.
    window = PointPolygonTestWindow( title = "Polygon Test" )
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
