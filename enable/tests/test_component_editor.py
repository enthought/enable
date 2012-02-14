""" Test the interaction between traitsui and enable's ComponentEditor.
"""


from enable.component_editor import ComponentEditor
from traits.has_traits import HasTraits
from traits.trait_types import Any
from traitsui.item import Item
from traitsui.view import View

from traitsui.tests._tools import *


class _ComponentDialog(HasTraits):
    """ View containing an item with ComponentEditor. """
    thing = Any

    traits_view = View(
        Item('thing', editor=ComponentEditor(), show_label=False),
        resizable = True
    )


DIALOG_WIDTH, DIALOG_HEIGHT = 700, 200
class _ComponentDialogWithSize(HasTraits):
    """ View containing an item with ComponentEditor and given size. """
    thing = Any

    traits_view = View(
        Item('thing', editor=ComponentEditor(), show_label=False,
             width=DIALOG_WIDTH, height=DIALOG_HEIGHT),
        resizable = True
    )


@skip_if_null
def test_initial_component():
    # BUG: the initial size of an Item with ComponentEditor is zero
    # in the Qt backend

    dialog = _ComponentDialog()
    ui = dialog.edit_traits()

    size = get_dialog_size(ui.control)
    nose.tools.assert_greater(size[0], 0)
    nose.tools.assert_greater(size[1], 0)



if __name__ == '__main__':
    # Executing the file opens the dialog for manual testing
    vw = _ComponentDialogWithSize()
    vw.configure_traits()

