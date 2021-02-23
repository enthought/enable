# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the PanTool class.
"""
# Enthought library imports
from enable.enable_traits import Pointer
from traits.api import Bool, Enum, Float, Tuple

from .drag_tool import DragTool


class ViewportPanTool(DragTool):
    """ A tool that enables the user to pan around a viewport by clicking a
    mouse button and dragging.
    """

    # The cursor to use when panning.
    drag_pointer = Pointer("hand")

    # Scaling factor on the panning "speed".
    speed = Float(1.0)

    # The modifier key that, if depressed when the drag is initiated,
    # constrains the panning to happen in the only direction of largest initial
    # motion. It is possible to permanently restrict this tool to always drag
    # along one direction.  To do so, set constrain=True, constrain_key=None,
    # and constrain_direction to the desired direction.
    constrain_key = Enum(None, "shift", "control", "alt")

    # Constrain the panning to one direction?
    constrain = Bool(False)

    # The direction of constrained draw. A value of None means that the user
    # has initiated the drag and pressed the constrain_key, but hasn't moved
    # the mouse yet; the magnitude of the components of the next mouse_move
    # event will determine the constrain_direction.
    constrain_direction = Enum(None, "x", "y")

    # (x,y) of the point where the mouse button was pressed.
    _original_xy = Tuple

    # Data coordinates of **_original_xy**.  This may be either (index,value)
    # or (value,index) depending on the component's orientation.
    _original_data = Tuple

    # Was constrain=True triggered by the **contrain_key**? If False, it was
    # set programmatically.
    _auto_constrain = Bool(False)

    # ------------------------------------------------------------------------
    # Inherited BaseTool traits
    # ------------------------------------------------------------------------

    # The tool is not visible (overrides BaseTool).
    visible = False

    def drag_start(self, event):
        self._original_xy = (event.x, event.y)
        if self.constrain_key is not None:
            if getattr(event, self.constrain_key + "_down"):
                self.constrain = True
                self._auto_constrain = True
                self.constrain_direction = None
        event.window.set_pointer(self.drag_pointer)
        event.window.set_mouse_owner(self, event.net_transform())
        event.handled = True

    def dragging(self, event):
        """ Handles the mouse being moved when the tool is in the 'panning'
        state.
        """
        if self._auto_constrain and self.constrain_direction is None:
            # Determine the constraint direction
            if (abs(event.x - self._original_xy[0])
                    > abs(event.y - self._original_xy[1])):
                self.constrain_direction = "x"
            else:
                self.constrain_direction = "y"

        new_position = self.component.view_position[:]
        for direction, ndx in [("x", 0), ("y", 1)]:
            if self.constrain and self.constrain_direction != direction:
                continue

            origpos = self._original_xy[ndx]
            eventpos = getattr(event, direction)
            delta = self.speed * (eventpos - origpos)
            if self.component.enable_zoom:
                delta /= self.component.zoom
            new_position[ndx] -= delta

        if self.constrain:
            _dir = self.constrain_direction
            self.component.view_position[_dir] = new_position[_dir]
        else:
            self.component.view_position = new_position
        event.handled = True

        self._original_xy = (event.x, event.y)
        self.component.request_redraw()

    def drag_end(self, event):
        if self._auto_constrain:
            self.constrain = False
            self.constrain_direction = None
        event.window.set_pointer("arrow")
        if event.window.mouse_owner == self:
            event.window.set_mouse_owner(None)
        event.handled = True
