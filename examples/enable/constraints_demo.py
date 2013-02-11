
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

        container.add(Component(id="one", bgcolor="red", fixed_preferred_size=(250,250)))
        container.add(Component(id="two", bgcolor="green", fixed_preferred_size=(250,250)))
        container.add(Component(id="three", bgcolor="blue", fixed_preferred_size=(250,250)))
        container.add(Component(id="four", bgcolor="white", fixed_preferred_size=(250,250)))

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
        ]

        return container


if __name__ == "__main__":
    Demo().configure_traits()
