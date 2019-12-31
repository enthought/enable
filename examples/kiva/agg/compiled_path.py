# CompiledPath should always be imported from the same backend as the
# GC you are using.  In this case, we are using the image GraphicsContext
# so we can save to disk when we're done, so we grab the CompiledPath
# from there as well.

import tempfile

from numpy import array
from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import ConstraintsContainer, Component, ComponentEditor
from enable.primitives.image import Image
from kiva.constants import STROKE
from kiva.image import GraphicsContext, CompiledPath


def compiled_path():
    # Creating the compiled path
    star_points = [(-20, -30),
                   (0, 30),
                   (20, -30),
                   (-30, 10),
                   (30, 10),
                   (-20, -30)]

    cross = CompiledPath()
    cross.scale_ctm(10.0, 10.0)
    lines = array(
        [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 2), (3, 2), (3, 1),
         (2, 1),
         (2, 0), (1, 0), (1, 1), (0, 1)])
    cross.lines(lines)

    gc = GraphicsContext((400, 400))
    gc.set_stroke_color((1, 0, 0, 1))
    gc.draw_path_at_points(array([(50, 50), (200, 50), (50, 200), (200, 200)]),
                           cross, STROKE)

    file_path = tempfile.mktemp(suffix='.jpg')

    gc.save(file_path)

    return file_path


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True)

    def _canvas_default(self):
        file_path = compiled_path()
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
