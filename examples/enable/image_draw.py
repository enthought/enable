"""
This demonstrates the use of the simple Image component.
"""
import os

from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import ConstraintsContainer, Component, ComponentEditor
from enable.primitives.image import Image

THIS_DIR = os.path.split(__file__)[0]


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True)

    def _canvas_default(self):
        path = os.path.join(THIS_DIR, 'deepfield.jpg')
        image = Image.from_file(path, resist_width='weak',
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
    Demo().configure_traits()
