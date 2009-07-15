import os.path
import xml.etree.cElementTree as etree

from enthought.enable.api import Container, Component, ComponentEditor, BaseTool
from enthought.kiva import Font, MODERN
from enthought.traits.api import Instance, Callable, List, Str, HasTraits
from enthought.traits.ui.api import View, Item
from enthought.savage.svg.document import SVGDocument
from enthought.savage.svg.backends.kiva.renderer import Renderer as KivaRenderer


class CanvasButton(Component):
    document = Instance(SVGDocument)
    label = Str()
    callback = Callable
    callback_args = List(Str)
    
    bounds = [64, 64]
    
    def __init__(self, filename, callback, callback_args, *args, **kw):
        super(CanvasButton, self).__init__(*args, **kw)
        
        if not os.path.exists(filename):
            raise ValueError
        tree = etree.parse(filename)
        root = tree.getroot()
        self.document =  SVGDocument(root, renderer=KivaRenderer)
        
        self.callback = callback
        self.callback_args = callback_args        
    
    def draw(self, gc, view_bounds, mode):
        gc.save_state()
        gc.translate_ctm(self.x, self.y+self.height)
        doc_size = self.document.getSize()
        gc.scale_ctm(self.width/float(doc_size[0]), -self.height/float(doc_size[1]))
        
        self.document.render(gc)
        gc.restore_state()
        
        if len(self.label) > 0:
            self._draw_label(gc)

        
    def _draw_label(self, gc):
        
        gc.save_state()

        font = Font(family=MODERN)
        gc.set_font(font)

        x, y, width, height = gc.get_text_extent(self.label)
        text_x = self.x + (self.width - width)/2.0
        text_y = self.y - height

        
        gc.show_text(self.label, (text_x, text_y))
        
        gc.restore_state()

    def perform(self):
        self.callback(*self.callback_args)
        

class ButtonCanvas(Container):
    def draw(self, gc, view_bounds=None, mode="default"):
        for component in self.components:
            component.draw(gc, view_bounds, mode)
            
    def add_button(self, button):
        button.container = self
        self.components.append(button)
        
class ButtonSelectionTool(BaseTool):
    """ Listens for double-clicks and tries to open a traits editor on the
        graph node under the mouse.
    """ 
        
    def normal_left_down(self, event):
        for component in self.component.components:
            if component.is_in(event.x, event.y):
                component.perform()
                break

        
class ButtonCanvasView(HasTraits):
    canvas = Instance(Container)
    
    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False),
                        width=400,
                        height=400,
                        resizable=True)
    
    def __init__(self, *args, **kw):
        super(ButtonCanvasView, self).__init__(*args, **kw)
        self.add_buttons()
    
    def _canvas_default(self):
        """ default setter for _canvas
        """
        container = ButtonCanvas()
        container.tools.append(ButtonSelectionTool(component=container))
        return container

    def add_buttons(self):
        data_dir = os.path.dirname(__file__)
        self.canvas.add_button(CanvasButton(os.path.join(data_dir, 'edit-copy.svg'), 
                                            self.do_copy, [], 
                                            label="Copy", x=150, y=150,))
        self.canvas.add_button(CanvasButton(os.path.join(data_dir, 'edit-paste.svg'), 
                                            self.do_paste, [], 
                                            label="Paste", x=250, y=150))

    def do_copy(self):
        print "copying something"
        
    def do_paste(self):
        print "pasting something"
        
        
if __name__ == "__main__":
    ButtonCanvasView().configure_traits()