from enable.savage.svg.document import SVGDocument
from enable.savage.trait_defs.ui.svg_editor import SVGEditor
from traits.api import HasTraits, Instance
from traitsui.api import Item, View


class StaticImageExample(HasTraits):
    svg = Instance(SVGDocument)

    traits_view = View(Item('svg', editor=SVGEditor(),
                            width=450, height=450,
                            show_label=False),
                       resizable=True,
                       title="StaticImageExample")

    def __init__(self, filename, renderer, *args, **kw):
        super(StaticImageExample, self).__init__(*args, **kw)

        self.svg = SVGDocument.createFromFile(filename, renderer=renderer)

if __name__ == "__main__":
    import os.path
    import sys

    if '--wx' in sys.argv:
        from enable.savage.svg.backends.wx.renderer import Renderer
        sys.argv.remove('--wx')
    elif '--kiva' in sys.argv:
        from enable.savage.svg.backends.kiva.renderer import Renderer
        sys.argv.remove('--kiva')
    else:
        from enable.savage.svg.backends.kiva.renderer import Renderer

    if len(sys.argv) > 1:
        StaticImageExample(sys.argv[1], Renderer).configure_traits()
    else:
        filename = os.path.join(os.path.dirname(__file__), 'lion.svg')
        StaticImageExample(filename, Renderer).configure_traits()
