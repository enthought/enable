"""
This demonstrates the `CtfEditor` gadget.

To use: right-click in the window to bring up a context menu. Once you've added
a color or opacity, you can drag them around by just clicking on them. The
colors at the end points are editable, but cannot be removed.

"""

from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt4'

from enable.api import ComponentEditor
from enable.gadgets.ctf_editor import CtfEditor
from enaml.colors import Color
from enaml.qt.qt_application import QtApplication
from traits.api import HasTraits, Instance
from traitsui.api import View, UItem
import traits_enaml

with traits_enaml.imports():
    from enaml.widgets.api import FileDialogEx
    from enaml.widgets.color_dialog import ColorDialog


def get_color(starting_color=None):
    """ Show a color picker to the user and return the color which is selected.
    """
    def color_as_tuple(color):
        return (color.red/255., color.green/255., color.blue/255.)

    def tuple_as_color(tup):
        r, g, b = tup
        return Color(int(r * 255), int(g * 255), int(b * 255), 255)

    dlg_kwargs = {'show_alpha': False}
    if starting_color:
        dlg_kwargs['current_color'] = tuple_as_color(starting_color)
    return color_as_tuple(ColorDialog.get_color(**dlg_kwargs))


def get_filename(action='save'):
    function = getattr(FileDialogEx, 'get_' + action + '_file_name')
    return function()


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
    ctf = CtfEditor(prompt_color_selection=get_color,
                    prompt_file_selection=get_filename)
    demo = Demo(ctf=ctf)

    app = QtApplication()
    demo.edit_traits()
    app.start()
