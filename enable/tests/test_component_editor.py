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


ITEM_WIDTH, ITEM_HEIGHT = 700, 200
class _ComponentDialogWithSize(HasTraits):
    """ View containing an item with ComponentEditor and given size. """
    thing = Any

    traits_view = View(
        Item('thing', editor=ComponentEditor(), show_label=False,
             width=ITEM_WIDTH, height=ITEM_HEIGHT),
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


@skip_if_null
def test_initial_component_with_item_size():
    # BEH: the initial component size should respect the size of the

    dialog = _ComponentDialogWithSize()
    ui = dialog.edit_traits()

    size = get_dialog_size(ui.control)

    # leave a few pixel of margin for wx
    nose.tools.assert_greater(size[0], ITEM_WIDTH-1)
    nose.tools.assert_less(size[0], ITEM_WIDTH+30)

    nose.tools.assert_greater(size[1], ITEM_HEIGHT-1)
    nose.tools.assert_less(size[1], ITEM_HEIGHT+30)



if __name__ == '__main__':
    # Executing the file opens the dialog for manual testing
    vw = _ComponentDialogWithSize()
    vw.configure_traits()

