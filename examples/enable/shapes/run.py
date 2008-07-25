""" An example showing moveable shapes. """


# Enthought library imports.
from enthought.enable.api import Window
from enthought.enable.example_support import DemoFrame, demo_main

# Local imports
from box import Box
from circle import Circle
from shape_container import ShapeContainer


class MyFrame(DemoFrame):
    """ The top-level frame. """
    
    ###########################################################################
    # 'DemoFrame' interface.
    ###########################################################################

    def _create_window(self):
        """ Create an enable window. """

        container = ShapeContainer(
            auto_size=False, bgcolor='black', *self._create_shapes()
        )

        return Window(self, component=container)

    ###########################################################################
    # Private interface.
    ###########################################################################

    def _create_shapes(self):
        """ Create some shapes. """

        box1 = Box(
            bounds     = [100, 100],
            position   = [50, 50],
            fill_color = 'red',
            text       = 'Box 1'
        )
        
        box2 = Box(
            bounds     = [100, 100],
            position   = [150, 150],
            fill_color = 'green',
            text       = 'Box 2'
        )
        
        circle1 = Circle(
            radius     = 50,
            position   = [250,250],
            fill_color = 'blue',
            text       = 'Circle 1'
        )

        circle2 = Circle(
            radius     = 50,
            position   = [350,350],
            fill_color = 'yellow',
            text       = 'Circle 2'
        )

        return box1, box2, circle1, circle2


if __name__ == "__main__":
    demo_main(MyFrame, size=(500, 500), title="Click and drag the shapes")

#### EOF ######################################################################
