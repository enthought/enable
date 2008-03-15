
from enthought.traits.api import Bool, Enum, Tuple

from drag_tool import DragTool


class ResizeTool(DragTool):
    """ Generic tool for resizing a component
    """

    drag_button = Enum("left", "right")

    # Should the moved component be raised to the top of its container's 
    # list of components?  This is only recommended for overlaying containers
    # and canvases, but generally those are the only ones in which the
    # ResizeTool will be useful.
    auto_raise = Bool(True)

    _corner = Enum("ul", "ur", "ll", "lr")
    _offset = Tuple(0, 0)

    # The last cursor position we saw; used during drag to compute deltas
    _prev_pos = Tuple(0, 0)

    def is_draggable(self, x, y):
        if self.component:
            c = self.component
            return (c.x <= x <= c.x2) and (c.y <= y <= c.y2)
        else:
            return False

    def drag_start(self, event):
        if self.component:
            self._prev_pos = (event.x, event.y)
            # Figure out which corner we are resizing
            if event.x > component.x + component.width/2:
                cx = 'r'  # right
                x_offset = component.x2 - event.x + 1
            else:
                cx = 'l'  # left
                x_offset = event.x - component.x + 1
            if event.y > component.y + component.height/2:
                cy = 'u'  # upper
                y_offset = component.y2 - event.y + 1
            else:
                cy = 'l'  # lower
                y_offset = event.y - component.y + 1
            self._corner = cy + cx
            self._offset = (x_offset, y_offset)
            self.component._layout_needed = True
            if self.auto_raise:
                # Push the component to the top of its container's list
                self.component.container.raise_component(self.component)
            event.window.set_mouse_owner(self, event.net_transform())
            event.handled = True
        return

    def dragging(self, event):
        if self.component:
            #dx = event.x - self._prev_pos[0]
            #dy = event.y - self._prev_pos[1]
            #pos = self.component.position
            #self.component.position = [pos[0] + dx, pos[1] + dy]
            offset = self._offset
            if self._corner[1] == 'l':   # left
                width
            else:                        # right
                x2 = event.x + offset

            self.component._layout_needed = True
            self.component.request_redraw()
            self._prev_pos = (event.x, event.y)
            event.handled = True
        return


