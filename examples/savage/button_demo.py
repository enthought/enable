from copy import copy
import os.path

from enable.savage.trait_defs.ui.svg_button import SVGButton
from traits.api import HasTraits, Str
from traitsui.api import Item, View, HGroup

button_size = (64, 64)

class ButtonDemo(HasTraits):

    copy_button = SVGButton(label='Copy',
                            filename=os.path.join(os.path.dirname(__file__),
                                                  'edit-copy.svg'),
                            width=button_size[0],
                            height=button_size[1])
    paste_button = SVGButton(label='Paste',
                             filename=os.path.join(os.path.dirname(__file__),
                                                   'edit-paste.svg'),
                             width=button_size[0],
                             height=button_size[1])
    text = Str
    clipboard = Str

    traits_view = View(HGroup(Item('copy_button', show_label=False),
                              Item('paste_button', show_label=False,
                                   enabled_when='len(clipboard)>0')),
                       Item('text', width=200),
                       title='SVG Button Demo')

    def _copy_button_fired(self, event):
        self.clipboard = copy(self.text)

    def _paste_button_fired(self, event):
        self.text += self.clipboard


if __name__ == "__main__":
    example = ButtonDemo()
    example.configure_traits()
