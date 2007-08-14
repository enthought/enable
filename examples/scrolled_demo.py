"""
Similar to simple_drag_demo, put one circle inside a scrolled container
"""
from numpy import array

from enthought.enable2.example_support import DemoFrame, demo_main

from enthought.enable2.api import Component, Scrolled, Container, Container,\
                                  Pointer, NativeScrollBar, Viewport, Window

from enthought.traits.api import Array, Enum, Float, Instance, Trait, Tuple


class Circle(Component):
    """
    The circle moves with the mouse cursor but leaves a translucent version of
    itself in its original position until the mouse button is released.
    """
    color = (0.0, 0.0, 1.0, 1.0)
    bgcolor = "none"
    
    normal_pointer = Pointer("arrow")
    moving_pointer = Pointer("hand")
    
    offset_x = Float
    offset_y = Float
    
    shadow_type = Enum("light", "dashed")
    shadow = Instance(Component)
    
    def __init__(self, **traits):
        Component.__init__(self, **traits)
        self.pointer = self.normal_pointer
        return
    
    def _draw(self, gc, view=None, mode="default"):
        gc.save_state()
        gc.set_fill_color(self.color)
        dx, dy = self.bounds
        x, y = self.position
        radius = min(dx/2.0, dy/2.0)
        gc.arc(x+dx/2.0, y+dy/2.0, radius, 0.0, 2*3.14159)
        gc.fill_path()
        return
    
    def normal_left_down(self, event):
        self.event_state = "moving"
        self.pointer = self.moving_pointer
        
        # Create our shadow
        if self.shadow_type == "light":
            klass = LightCircle
        else:
            klass = DashedCircle
        dx, dy = self.bounds
        self.shadow = klass(bounds=self.bounds, position=self.position,
                            color=self.color)
        self.container.add(self.shadow)
        x, y = self.position
        self.offset_x = event.x - x
        self.offset_y = event.y - y
        return

    def moving_mouse_move(self, event):
        self.position = [event.x-self.offset_x, event.y-self.offset_y]
        self.request_redraw()
        return

    def moving_left_up(self, event):
        self.event_state = "normal"
        self.pointer = self.normal_pointer
        self.request_redraw()
        # Remove our shadow
        self.container.remove(self.shadow)
        return

    def moving_mouse_leave(self, event):
        self.moving_left_up(event)
        return


class LightCircle(Component):
    
    color = Tuple
    bgcolor = "none"
    radius = Float(1.0)
    
    def _draw(self, gc, view=None, mode="default"):
        gc.save_state()
        gc.set_fill_color(self.color[0:3] + (self.color[3]*0.3,))
        dx, dy = self.bounds
        x, y = self.position
        radius = min(dx/2.0, dy/2.0)
        gc.arc(x+dx/2.0, y+dy/2.0, radius, 0.0, 2*3.14159)
        gc.fill_path()
        gc.restore_state()
        return

class DashedCircle(Component):
    
    color = Tuple
    bgcolor = "none"
    radius = Float(1.0)
    line_dash = array([2.0, 2.0])
    
    def _draw(self, gc, view=None, mode="default"):
        gc.save_state()
        gc.set_fill_color(self.color)
        dx, dy = self.bounds
        x, y = self.position
        radius = min(dx/2.0, dy/2.0)
        gc.arc(x+dx/2.0, y+dy/2.0, radius, 0.0, 2*3.14159)
        gc.set_stroke_color(self.color[0:3] + (self.color[3]*0.8,))
        gc.set_line_dash(self.line_dash)
        gc.stroke_path()
        gc.restore_state()
        return


class MyFrame(DemoFrame):
    def _create_window(self):

        circle1 = Circle(bounds=[75,75], position=[100,100], shadow_type="dashed")
        container = Container(bounds=[300,300], bgcolor="red",
                                     auto_size = False)
        scr = Scrolled(container, bounds=[200,200], position=[50,50])
        container.add(circle1)
        
        scr2 = Scrolled(container, bounds=[200,200], position=[300,50])
        topcontainer = Container()
        topcontainer.add(scr)
        topcontainer.add(scr2)
        return Window(self, -1, component=topcontainer)


if __name__ == "__main__":
    demo_main(MyFrame, title="Click and drag to move the circles")
    
# EOF
