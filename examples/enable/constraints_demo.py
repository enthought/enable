
from enable.api import Component, ComponentEditor, ConstraintsContainer
from enable.layout.layout_helpers import hbox, vbox, align, grid, vertical
from traits.api import HasTraits, Any, Instance, List, Property
from traitsui.api import Item, View, HGroup, TabularEditor
from traitsui.tabular_adapter import TabularAdapter


class ConstraintAdapter(TabularAdapter):
    """ Display Constraints in a TabularEditor.
    """
    columns = [('Constraint', 'id')]
    id_text = Property
    def _get_id_text(self):
        return self.item.__repr__()


class Demo(HasTraits):
    canvas = Instance(Component)

    constraints = Property(List)

    selected_constraints = Any

    traits_view = View(
                        HGroup(
                            Item('constraints',
                                 editor=TabularEditor(
                                    adapter=ConstraintAdapter(),
                                    editable=False,
                                    multi_select=True,
                                    selected='selected_constraints',
                                 ),
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
        parent = ConstraintsContainer(bounds=(500,500), padding=20, debug=True)

        hugs = {'hug_width':'weak', 'hug_height':'weak'}
        one = Component(id="one", bgcolor=0xFF0000, **hugs)
        two = Component(id="two", bgcolor=0x00FF00, **hugs)
        three = Component(id="three", bgcolor=0x0000FF, **hugs)
        four = Component(id="four", bgcolor=0x000000, **hugs)

        parent.add(one, two, three, four)
        parent.layout_constraints = [
            grid([one.constraints, two.constraints],
                 [three.constraints, four.constraints]),
            align('height', one.constraints, two.constraints,
                  three.constraints, four.constraints),
            align('width', one.constraints, two.constraints,
                  three.constraints, four.constraints),
        ]

        return parent

    def _get_constraints(self):
        if self.canvas._layout_manager._constraints:
            return list(self.canvas._layout_manager._constraints)
        return []

    def _selected_constraints_changed(self, new):
        if new is None or new == []:
            return

        if self.canvas.debug:
            canvas = self.canvas
            canvas._debug_overlay.selected_constraints = new
            canvas.request_redraw()


if __name__ == "__main__":
    Demo().configure_traits()
