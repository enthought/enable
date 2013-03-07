""" An example showing moveable shapes. """
# Enthought library imports.
from enable.api import Container, Window
from enable.example_support import DemoFrame, demo_main

# Local imports
from box import Box
from circle import Circle


class MyFrame(DemoFrame):
    """ The top-level frame. """

    # 'DemoFrame' interface.
    #--------------------------------------------------------------------------

    def _create_window(self):
        """ Create an enable window. """

        container = Container(
            auto_size=False, bgcolor='black', *self._create_shapes()
        )

        return Window(self, component=container)

    # Private interface.
    #--------------------------------------------------------------------------

    def _create_shapes(self):
        """ Create some shapes. """

        box1 = Box(
            bounds     = [100, 100],
            position   = [50, 50],
            fill_color = 'lightpink',
            text       = 'Box 1'
        )

        box2 = Box(
            bounds     = [100, 100],
            position   = [150, 150],
            fill_color = 'greenyellow',
            text       = 'Box 2'
        )

        circle1 = Circle(
            radius     = 50,
            position   = [250,250],
            fill_color = 'cornflowerblue',
            text       = 'Circle 1'
        )

        circle2 = Circle(
            radius     = 50,
            position   = [350,350],
            fill_color = 'khaki',
            text       = 'Circle 2'
        )

        return box1, box2, circle1, circle2


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(MyFrame, size=(500, 500),
                     title="Click and drag the shapes")
