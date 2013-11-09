"""
This allows a simple component to be moved around the screen.
"""
from enable.example_support import DemoFrame, demo_main

from traits.api import Float
from enable.api import Component, Pointer, Window


class Box(Component):
    """
    The box moves wherever the user clicks and drags.
    """
    normal_pointer = Pointer("arrow")
    moving_pointer = Pointer("hand")

    offset_x = Float
    offset_y = Float

    fill_color = (0.8, 0.0, 0.1, 1.0)
    moving_color = (0.0, 0.8, 0.1, 1.0)

    resizable = ""

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        with gc:
            if self.event_state == "moving":
                gc.set_fill_color(self.moving_color)
            else:
                gc.set_fill_color(self.fill_color)
            dx, dy = self.bounds
            x, y = self.position
            gc.rect(x, y, dx, dy)
            gc.fill_path()

            # draw line around outer box
            gc.set_stroke_color((0,0,0,1))
            gc.rect(self.outer_x, self.outer_y, self.outer_width,
                    self.outer_height)
            gc.stroke_path()

        return

    def normal_key_pressed(self, event):
        print("Key:", event.character)

    def normal_left_down(self, event):
        self.event_state = "moving"
        event.window.set_pointer(self.moving_pointer)
        event.window.set_mouse_owner(self, event.net_transform())
        self.offset_x = event.x - self.x
        self.offset_y = event.y - self.y
        event.handled = True
        return

    def moving_mouse_move(self, event):
        self.position = [event.x-self.offset_x, event.y-self.offset_y]
        event.handled = True
        self.request_redraw()
        return

    def moving_left_up(self, event):
        self.event_state = "normal"
        event.window.set_pointer(self.normal_pointer)
        event.window.set_mouse_owner(None)
        event.handled = True
        self.request_redraw()
        return

    def moving_mouse_leave(self, event):
        self.moving_left_up(event)
        event.handled = True
        return


class MyFrame(DemoFrame):

    def _create_window(self):
        box = Box(bounds=[100,100], position=[50,50], padding=15)
        return Window(self, -1, component=box)


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(MyFrame, title="Click and drag to move the box")
