import tempfile

from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import ConstraintsContainer, Component, ComponentEditor
from enable.kiva_graphics_context import GraphicsContext
from enable.primitives.image import Image
from kiva import constants
from kiva.image import GraphicsContext


def simple():
    gc = GraphicsContext((100, 100))

    gc.clear()
    gc.set_line_cap(constants.CAP_SQUARE)
    gc.set_line_join(constants.JOIN_MITER)
    gc.set_stroke_color((1, 0, 0))
    gc.set_fill_color((0, 0, 1))
    gc.rect(0, 0, 30, 30)
    gc.draw_path()
    file_path = tempfile.mktemp(suffix='.bmp')
    gc.save(file_path)
    return file_path


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True)

    def _canvas_default(self):
        file_path = simple()
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
