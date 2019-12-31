import tempfile

from numpy import array
from scipy import pi
from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import ConstraintsContainer, Component, ComponentEditor
from enable.kiva_graphics_context import GraphicsContext
from enable.primitives.image import Image
from kiva import constants
from kiva.image import GraphicsContext


def draw(gc):
    # colors are 5 doubles: offset, red, green, blue, alpha
    starting_color = array([0.0, 1.0, 1.0, 1.0, 1.0])
    ending_color = array([1.0, 0.0, 0.0, 0.0, 1.0])

    gc.clear()

    # diagonal
    with gc:
        gc.rect(50, 25, 150, 100)
        gc.linear_gradient(50, 25, 150, 125,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)

    # vertical top to bottom
    with gc:
        gc.rect(50, 150, 150, 50)
        gc.linear_gradient(0, 200, 0, 150,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)
    # horizontal left to right
    with gc:
        gc.rect(50, 200, 150, 50)
        gc.linear_gradient(50, 0, 150, 0,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)

    # vertical bottom to top
    with gc:
        gc.rect(50, 275, 150, 50)
        gc.linear_gradient(0, 275, 0, 325,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)
    # horizontal right to left
    with gc:
        gc.rect(50, 325, 150, 50)
        gc.linear_gradient(200, 0, 100, 0,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)

    # radial
    with gc:
        gc.arc(325, 75, 50, 0.0, 2 * pi)
        gc.radial_gradient(325, 75, 50, 325, 75,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)

    # radial with focal point in upper left
    with gc:
        gc.arc(325, 200, 50, 0.0, 2 * pi)
        gc.radial_gradient(325, 200, 50, 300, 225,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)

    # radial with focal point in bottom right
    with gc:
        gc.arc(325, 325, 50, 0.0, 2 * pi)
        gc.radial_gradient(325, 325, 50, 350, 300,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)

    return


def gradient():
    gc = GraphicsContext((500, 500))
    gc.scale_ctm(1.25, 1.25)
    draw(gc)
    file_path = tempfile.mktemp(suffix='.png')
    gc.save(file_path, file_format='png')
    return file_path


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True)

    def _canvas_default(self):
        file_path = gradient()
        image = Image.from_file(file_path, resist_width='weak',
                                resist_height='weak')

        container = ConstraintsContainer(bounds=[500, 500])
        container.add(image)
        ratio = float(image.data.shape[1]) / image.data.shape[0]
        container.layout_constraints = [
            image.left == container.contents_left,
            image.right == container.contents_right,
            image.top == container.contents_top,
            image.bottom == container.contents_bottom,
            image.layout_width == ratio * image.layout_height,
        ]

        return container


if __name__ == '__main__':
    Demo().configure_traits()
