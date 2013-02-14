
from enable.api import Component, ComponentEditor, ConstraintsContainer
from enable.layout.layout_helpers import hbox, align, grid
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
            hbox(one.constraints, two.constraints,
                 three.constraints, four.constraints),
            align('width', one.constraints, two.constraints,
                           three.constraints, four.constraints),

        ]

        return parent


if __name__ == "__main__":
    Demo().configure_traits()
