
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
        parent = ConstraintsContainer(bounds=(500,500))

        hugs = {'hug_width':'weak', 'hug_height':'weak'}
        one = Component(id="one", bgcolor=0xFF0000, **hugs)
        two = Component(id="two", bgcolor=0x00FF00, **hugs)
        three = Component(id="three", bgcolor=0x0000FF, **hugs)
        four = Component(id="four", bgcolor=0x000000, **hugs)

        parent.add(one, two, three, four)
        parent.layout_constraints = [
            parent.constraints.top == one.constraints.top,
            parent.constraints.left == one.constraints.left,
            parent.constraints.top == two.constraints.top,
            parent.constraints.right == two.constraints.right,
            parent.constraints.bottom == three.constraints.bottom,
            parent.constraints.left == three.constraints.left,
            parent.constraints.bottom == four.constraints.bottom,
            parent.constraints.right == four.constraints.right,
            one.constraints.right == two.constraints.left,
            three.constraints.right == four.constraints.left,
            one.constraints.bottom == three.constraints.top,
            two.constraints.bottom == four.constraints.top,
            one.constraints.width == two.constraints.width,
            one.constraints.width == three.constraints.width,
            one.constraints.width == four.constraints.width,
            one.constraints.height == three.constraints.height,
            one.constraints.height == two.constraints.height,
            one.constraints.height == four.constraints.height,
        ]

        return parent


if __name__ == "__main__":
    Demo().configure_traits()
