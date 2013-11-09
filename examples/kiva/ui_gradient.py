

import numpy as np

from enable.api import Component, ComponentEditor
from traits.api import HasTraits, Instance
from traitsui.api import Item, View

class MyCanvas(Component):
    def draw(self, gc, **kwargs):
        w,h = gc.width(), gc.height()

        # colors are 5 doubles: offset, red, green, blue, alpha
        starting_color = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        ending_color = np.array([1.0, 0.0, 0.0, 0.0, 1.0])

        gc.clear()

        # radial reflected background
        with gc:
            gc.rect(0, 0, w, h)

            start = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
            end = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
            gc.radial_gradient(w/4, h/4, 200, w/4+100, h/4+100,
                               np.array([start, end]), 'reflect')
            gc.fill_path()

        # diagonal
        with gc:
            gc.rect(50,25,150,100)
            gc.linear_gradient(0,0,1,1,
                                np.array([starting_color, ending_color]),
                                "pad", 'objectBoundingBox')
            gc.fill_path()

        # vertical
        with gc:
            gc.rect(50,150,150,100)
            gc.linear_gradient(50,150,50,250,
                                np.array([starting_color, ending_color]),
                                "pad", 'userSpaceOnUse')
            gc.fill_path()

        # horizontal
        with gc:
            gc.rect(50,275,150,100)
            gc.linear_gradient(0,0,1,0,
                                np.array([starting_color, ending_color]),
                                "pad", 'objectBoundingBox')
            gc.fill_path()

        # radial
        with gc:
            gc.arc(325, 75, 50, 0.0, 2*np.pi)
            gc.radial_gradient(325, 75, 50, 325, 75,
                                np.array([starting_color, ending_color]),
                                "pad", 'userSpaceOnUse')
            gc.fill_path()

        # radial with focal point in upper left
        with gc:
            gc.arc(325, 200, 50, 0.0, 2*np.pi)
            gc.radial_gradient(0.5, 0.5, 0.5, 0.25, 0.75,
                            np.array([starting_color, ending_color]),
                            "pad", 'objectBoundingBox')
            gc.fill_path()

        # radial with focal point in bottom right
        with gc:
            gc.arc(325, 325, 50, 0.0, 2*np.pi)
            gc.radial_gradient(325, 325, 50, 350, 300,
                                np.array([starting_color, ending_color]),
                                "pad", 'userSpaceOnUse')
            gc.fill_path()

        return

class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(bgcolor="lightgray"),
                            show_label=False, width=500, height=500),
                       resizable=True, title="Gradient Example")

    def _canvas_default(self):
        return MyCanvas()


if __name__ == "__main__":
    Demo().configure_traits()
