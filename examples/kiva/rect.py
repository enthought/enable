import tempfile

from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import ConstraintsContainer, Component, ComponentEditor
from enable.kiva_graphics_context import GraphicsContext
from enable.primitives.image import Image
from kiva.image import GraphicsContext


def rect():
    gc = GraphicsContext((500, 500))
    gc.clear()
    gc.rect(100, 100, 300, 300)
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
        file_path = rect()
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
