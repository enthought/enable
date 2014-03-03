"""
This demonstrates the `CtfEditor` gadget.

To use: right-click in the window to bring up a context menu. Once you've added
a color or opacity, you can drag them around by just clicking on them. The
colors at the end points are editable, but cannot be removed.

"""

from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt4'

from enaml.qt.qt_application import QtApplication
from traits.api import HasTraits, Instance
from traitsui.api import View, UItem
from enable.api import ComponentEditor

from enable.gadgets.ctf_editor import CtfEditor


class Demo(HasTraits):

    ctf = Instance(CtfEditor)

    traits_view = View(
        UItem('ctf',
              editor=ComponentEditor(),
              style='custom'),
        width=450,
        height=150,
        title="Color Transfer Function Editor",
        resizable=True,
    )


if __name__ == "__main__":
    ctf = CtfEditor()
    demo = Demo(ctf=ctf)

    app = QtApplication()
    demo.edit_traits()
    app.start()
