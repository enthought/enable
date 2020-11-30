import tempfile

from enable.api import ConstraintsContainer
from enable.example_support import DemoFrame, demo_main
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


class Demo(DemoFrame):
    def _create_component(self):
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


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo)
