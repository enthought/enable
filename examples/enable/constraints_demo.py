
from enable.api import Component, ComponentEditor, ConstraintsContainer
from enable.layout.layout_helpers import (align, grid, horizontal, hbox, vbox,
    spacer, vertical)
from traits.api import HasTraits, Instance, Str
from traitsui.api import Item, View, HGroup, CodeEditor


class Demo(HasTraits):
    canvas = Instance(Component)

    constraints_def = Str

    traits_view = View(
                        HGroup(
                            Item('constraints_def',
                                 editor=CodeEditor(),
                                 height=100,
                                 show_label=False,
                            ),
                            Item('canvas',
                                 editor=ComponentEditor(),
                                 show_label=False,
                            ),
                        ),
                        resizable=True,
                        title="Constraints Demo",
                        width=1000,
                        height=500,
                 )

    def _canvas_default(self):
        parent = ConstraintsContainer(bounds=(500,500), padding=20)

        hugs = {'hug_width':'weak', 'hug_height':'weak'}
        one = Component(id="one", bgcolor=0xFF0000, **hugs)
        two = Component(id="two", bgcolor=0x00FF00, **hugs)
        three = Component(id="three", bgcolor=0x0000FF, **hugs)
        four = Component(id="four", bgcolor=0x000000, **hugs)

        parent.add(one, two, three, four)
        return parent

    def _constraints_def_changed(self):
        if self.canvas is None:
            return

        canvas = self.canvas
        components = canvas._components
        one = components[0]
        two = components[1]
        three = components[2]
        four = components[3]

        try:
            new_cns = eval(self.constraints_def)
        except Exception, ex:
            return

        self.canvas.layout_constraints = new_cns
        self.canvas.request_redraw()

    def _constraints_def_default(self):
        return """[
    grid([one, two], [three, four]),
    align('layout_height', one, two, three, four),
    align('layout_width', one, two, three, four),
]"""


if __name__ == "__main__":
    demo = Demo()
    demo._constraints_def_changed()
    demo.configure_traits()
