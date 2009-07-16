from enthought.savage.svg.document import SVGDocument
from enthought.savage.traits.ui.svg_editor import SVGEditor
from enthought.traits.api import HasTraits, Instance
from enthought.traits.ui.api import Item, View
import xml.etree.cElementTree as etree

from enthought.savage.svg.backends.wx.renderer import Renderer as WxRenderer
from enthought.savage.svg.backends.kiva.renderer import Renderer as KivaRenderer

class StaticImageExample(HasTraits):
    svg = Instance(SVGDocument)
    
    traits_view = View(Item('svg', editor=SVGEditor(), 
                            width=350, height=450,
                            show_label=False),
                       title="StaticImageExample")
                            
    def __init__(self, filename, renderer, *args, **kw):
        super(StaticImageExample, self).__init__(*args, **kw)

        self.svg = SVGDocument.createFromFile(filename, renderer=renderer)
        
if __name__ == "__main__":
    import os.path
    import sys

    renderer = WxRenderer

    if '--wx' in sys.argv:
        renderer = WxRenderer
        sys.argv.remove('--wx')
    if '--kiva' in sys.argv:
        renderer = KivaRenderer
        sys.argv.remove('--kiva')

    if len(sys.argv) > 1:
        StaticImageExample(sys.argv[1], renderer).configure_traits()
    else:
        filename = os.path.join(os.path.dirname(__file__), 'lion.svg')
        StaticImageExample(filename, renderer).configure_traits()
     
