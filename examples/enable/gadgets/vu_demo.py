from traits.api import HasTraits, Instance
from traitsui.api import View, UItem, Item, RangeEditor, Group, VGroup
from enable.api import ComponentEditor

from enable.gadgets.vu_meter import VUMeter


class Demo(HasTraits):

    vu = Instance(VUMeter)

    traits_view = View(
        VGroup(
            Group(
                UItem('vu', editor=ComponentEditor(size=(60, 60)),
                             style='custom'),
            ),
            Item('object.vu.percent',
                 editor=RangeEditor(low=0.0, high=200.0, mode='slider')),
        ),
        '_',
        VGroup(
            Item('object.vu.angle', label="angle",
                 editor=RangeEditor(low=0.0, high=89.0, mode='slider')),
            Item('object.vu._beta',
                 editor=RangeEditor(low=0.0, high=1.0, mode='slider')),
        ),
        width=450,
        height=380,
        title="VU Meter",
        resizable=True,
    )


if __name__ == "__main__":
    color = (0.9, 0.85, 0.7)
    vu = VUMeter(border_visible=True, border_width=2, bgcolor=color)

    demo = Demo(vu=vu)
    demo.configure_traits()
