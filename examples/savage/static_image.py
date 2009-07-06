from enthought.savage.svg.document import SVGDocument
from enthought.savage.traits.ui.svg_editor import SVGEditor
from enthought.traits.api import HasTraits, Instance
from enthought.traits.ui.api import Item, View
import xml.etree.cElementTree as etree

class StaticImageExample(HasTraits):
    svg = Instance(SVGDocument)
    
    traits_view = View(Item('svg', editor=SVGEditor(), 
                            width=350, height=450,
                            show_label=False),
                       title="StaticImageExample")
                            
    def __init__(self, filename, *args, **kw):
        super(StaticImageExample, self).__init__(*args, **kw)
        
        tree = etree.parse(filename)
        root = tree.getroot()
        
        # FIXME: programatically figure out which renderer to use
        from enthought.savage.svg.backends.wx.renderer import Renderer as WxRenderer
        from enthought.savage.svg.backends.kiva.renderer import Renderer as KivaRenderer
        self.svg = SVGDocument(root, renderer=KivaRenderer)
        
if __name__ == "__main__":
    import os.path
    sample_file = os.path.join(os.path.dirname(__file__), 'lion.svg')
    example = StaticImageExample(sample_file)
    example.configure_traits()
