"""
Basic demo of drawing within an Enable component.
"""
from enable.api import Component, ComponentEditor
from traits.api import HasTraits, Instance
from traitsui.api import Item, View


class MyComponent(Component):
    def draw(self, gc, **kwargs):
        w,h = gc.width(), gc.height()
        gc.clear()

        # Draw a rounded rect just inside the bounds
        gc.set_line_width(2.0)
        gc.set_stroke_color((0.0, 0.0, 0.0, 1.0))

        r = 15
        b = 3
        gc.move_to(b, h/2)
        gc.arc_to(b, h-b,
                  w/2, h-b,
                  r)
        gc.arc_to(w-b, h-b,
                  w-b, h/2,
                  r)
        gc.arc_to(w-b, b,
                  w/2, b,
                  r)
        gc.arc_to(b, b,
                  b, h/2,
                  r)
        gc.line_to(b, h/2)
        gc.stroke_path()
        return

    def normal_key_pressed(self, event):
        print("key pressed: ", event.character)


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True, title="Component Example")

    def _canvas_default(self):
        return MyComponent()


if __name__ == "__main__":
    Demo().configure_traits()
