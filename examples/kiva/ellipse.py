import tempfile

from scipy import pi

from enable.api import ConstraintsContainer
from enable.example_support import DemoFrame, demo_main
from enable.primitives.image import Image
from kiva.image import GraphicsContext


def draw_ellipse(gc, x, y, major, minor, angle):
    """ Draws an ellipse given major and minor axis lengths.  **angle** is
    the angle between the major axis and the X axis, in radians.
    """
    with gc:
        gc.translate_ctm(x, y)
        ratio = float(major) / minor
        gc.rotate_ctm(angle)
        gc.scale_ctm(ratio, 1.0)
        gc.arc(0, 0, minor, 0.0, 2 * pi)
        gc.stroke_path()
        gc.move_to(-minor, 0)
        gc.line_to(minor, 0)
        gc.move_to(0, -minor)
        gc.line_to(0, minor)
        gc.stroke_path()


def ellipse():
    gc = GraphicsContext((300, 300))
    gc.set_alpha(0.3)
    gc.set_stroke_color((1.0, 0.0, 0.0))
    gc.set_fill_color((0.0, 1.0, 0.0))
    gc.rect(95, 95, 10, 10)
    gc.fill_path()
    draw_ellipse(gc, 100, 100, 35.0, 25.0, pi / 6)
    file_path = tempfile.mktemp(suffix='.bmp')
    gc.save(file_path)
    return file_path


class Demo(DemoFrame):

    def _create_component(self):
        file_path = ellipse()
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
