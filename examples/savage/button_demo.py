import os.path

from enthought.savage.traits.ui.wx.svg_button import SVGButton
from enthought.savage.traits.ui.wx.svg_button_editor import SVGButtonEditor
from enthought.traits.api import HasTraits, Instance, Str
from enthought.traits.ui.api import Item, View, HGroup
import xml.etree.cElementTree as etree

class ButtonDemo(HasTraits):
    copy_button = SVGButton('copy', filename=os.path.join(os.path.dirname(__file__), 'edit-copy.svg'), width=32, height=32)
    paste_button = SVGButton('paste', filename=os.path.join(os.path.dirname(__file__), 'edit-paste.svg'))
    text = Str()
    
    traits_view = View(HGroup(Item('copy_button', width=32, height=32, show_label=False),
                              Item('paste_button', width=32, height=32, show_label=False)),
                       Item('text'),
                       title="SVG Button Demo")
                                    
if __name__ == "__main__":
    import os.path
    example = ButtonDemo()
    example.configure_traits()
