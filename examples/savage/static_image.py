from enthought.savage.svg.document import SVGDocument
from enthought.savage.traits.ui.svg_editor import SVGEditor
from enthought.traits.api import HasTraits, Instance
from enthought.traits.ui.api import Item, View

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

    if '--wx' in sys.argv:
        from enthought.savage.svg.backends.wx.renderer import Renderer
        sys.argv.remove('--wx')
    elif '--kiva' in sys.argv:
        from enthought.savage.svg.backends.kiva.renderer import Renderer 
        sys.argv.remove('--kiva')
    else:
        from enthought.savage.svg.backends.kiva.renderer import Renderer

    if len(sys.argv) > 1:
        StaticImageExample(sys.argv[1], Renderer).configure_traits()
    else:
        filename = os.path.join(os.path.dirname(__file__), 'lion.svg')
        StaticImageExample(filename, Renderer).configure_traits()
     
