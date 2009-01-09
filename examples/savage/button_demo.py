import copy
import os.path

from enthought.savage.traits.ui.wx.svg_button import SVGButton
from enthought.savage.traits.ui.wx.svg_button_editor import SVGButtonEditor
from enthought.traits.api import HasTraits, Instance, Str, Int
from enthought.traits.ui.api import Item, View, HGroup

button_size = (128, 128)

class ButtonDemo(HasTraits):
    
    copy_button = SVGButton('copy', filename=os.path.join(os.path.dirname(__file__), 'edit-copy.svg'), 
                            width=button_size[0], height=button_size[1])
    paste_button = SVGButton('paste', filename=os.path.join(os.path.dirname(__file__), 'edit-paste.svg'), 
                            width=button_size[0], height=button_size[1])
    text = Str()
    
    clipboard = Str()
    
    traits_view = View(HGroup(Item('copy_button', show_label=False),
                              Item('paste_button', show_label=False)),
                       Item('text', width=200),
                       title="SVG Button Demo")
                       
    def _copy_button_fired(self, event):
        self.clipboard = copy.copy(self.text)
        
    def _paste_button_fired(self, event):
        self.text += self.clipboard
                                    
if __name__ == "__main__":
    import os.path
    example = ButtonDemo()
    example.configure_traits()
