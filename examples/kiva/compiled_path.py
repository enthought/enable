# CompiledPath should always be imported from the same backend as the
# GC you are using.  In this case, we are using the image GraphicsContext
# so we can save to disk when we're done, so we grab the CompiledPath
# from there as well.

import tempfile

from enable.api import ConstraintsContainer
from enable.example_support import DemoFrame, demo_main
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

    path = CompiledPath()
    path.move_to(*star_points[0])
    for pt in star_points[1:]:
        path.line_to(*pt)

    locs = [(100, 100), (100, 300), (100, 500), (200, 100), (200, 300),
            (200, 500)]

    gc = GraphicsContext((300, 600))
    gc.set_stroke_color((0, 0, 1, 1))
    gc.draw_path_at_points(locs, path, STROKE)

    file_path = tempfile.mktemp(suffix='.jpg')

    gc.save(file_path)

    return file_path


class Demo(DemoFrame):

    def _create_component(self):
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


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo)
