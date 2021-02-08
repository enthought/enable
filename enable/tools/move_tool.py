# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from traits.api import Bool, Enum, Tuple

from .drag_tool import DragTool


class MoveTool(DragTool):
    """ Generic tool for moving a component's position relative to its
    container
    """

    drag_button = Enum("left", "right")

    # Should the moved component be raised to the top of its container's
    # list of components?  This is only recommended for overlaying containers
    # and canvases, but generally those are the only ones in which the
    # MoveTool will be useful.
    auto_raise = Bool(True)

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
            self.component._layout_needed = True
            if self.auto_raise:
                # Push the component to the top of its container's list
                self.component.container.raise_component(self.component)
            event.window.set_mouse_owner(self, event.net_transform())
            event.handled = True

    def dragging(self, event):
        if self.component:
            dx = event.x - self._prev_pos[0]
            dy = event.y - self._prev_pos[1]
            pos = self.component.position
            self.component.position = [pos[0] + dx, pos[1] + dy]
            self.component._layout_needed = True
            self.component.request_redraw()
            self._prev_pos = (event.x, event.y)
            event.handled = True
