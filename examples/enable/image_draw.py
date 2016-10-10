"""
This demonstrates the use of the simple Image component.
"""
import os

from enable.api import ConstraintsContainer, Window
from enable.example_support import DemoFrame, demo_main
from enable.primitives.image import Image

THIS_DIR = os.path.split(__file__)[0]


class MyFrame(DemoFrame):

    def _create_window(self):
        path = os.path.join(THIS_DIR, 'deepfield.jpg')
        image = Image.from_file(path, resist_width='weak',
                                resist_height='weak')

        container = ConstraintsContainer(bounds=[500, 500])
        container.add(image)
        ratio = float(image.data.shape[1])/image.data.shape[0]
        container.layout_constraints = [
            image.left == container.contents_left,
            image.right == container.contents_right,
            image.top == container.contents_top,
            image.bottom == container.contents_bottom,
            image.layout_width == ratio*image.layout_height,
        ]
        return Window(self, -1, component=container)


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(MyFrame)
