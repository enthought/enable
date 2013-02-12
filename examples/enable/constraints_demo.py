
from enable.api import Component, ComponentEditor, ConstraintsContainer
from traits.api import HasTraits, Instance
from traitsui.api import Item, View


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(
                        Item('canvas',
                             editor=ComponentEditor(),
                             show_label=False,
                        ),
                        resizable=True,
                        title="Constraints Demo",
                        width=500,
                        height=500,
                 )

    def _canvas_default(self):
        container = ConstraintsContainer(bounds=(500,500))

        hugs = {'hug_width':'weak', 'hug_height':'weak'}
        container.add(Component(id="one", bgcolor=0xFF0000, **hugs))
        container.add(Component(id="two", bgcolor=0x00FF00, **hugs))
        container.add(Component(id="three", bgcolor=0x0000FF, **hugs))
        container.add(Component(id="four", bgcolor=0x000000, **hugs))

        container.layout_constraints = [
            "parent.top == one.top",
            "parent.left == one.left",
            "parent.top == two.top",
            "parent.right == two.right",
            "parent.bottom == three.bottom",
            "parent.left == three.left",
            "parent.bottom == four.bottom",
            "parent.right == four.right",
            "one.right == two.left",
            "three.right == four.left",
            "one.bottom == three.top",
            "two.bottom == four.top",
            "one.width == two.width",
            "one.width == three.width",
            "one.width == four.width",
            "one.height == three.height",
            "one.height == two.height",
            "one.height == four.height",
        ]

        return container


if __name__ == "__main__":
    Demo().configure_traits()
