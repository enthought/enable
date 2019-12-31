import tempfile
from math import pi

from scipy import pi
from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import ConstraintsContainer, Component, ComponentEditor
from enable.primitives.image import Image
from kiva import agg


def add_star(gc):
    gc.begin_path()
    gc.move_to(-20, -30)
    gc.line_to(0, 30)
    gc.line_to(20, -30)
    gc.line_to(-30, 10)
    gc.line_to(30, 10)
    gc.close_path()
    gc.move_to(-10, 30)
    gc.line_to(10, 30)


def stars():
    gc = agg.GraphicsContextArray((500, 500))

    gc.set_alpha(0.3)
    gc.set_stroke_color((1.0, 0.0, 0.0))
    gc.set_fill_color((0.0, 1.0, 0.0))

    for i in range(0, 600, 5):
        with gc:
            gc.translate_ctm(i, i)
            gc.rotate_ctm(i * pi / 180.)
            add_star(gc)
            gc.draw_path()

    gc.set_fill_color((0.5, 0.5, 0.5))
    gc.rect(150, 150, 200, 200)
    gc.fill_path()
    file_path = tempfile.mktemp(suffix='.bmp')
    gc.save(file_path)
    return file_path


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True)

    def _canvas_default(self):
        file_path = stars()
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
