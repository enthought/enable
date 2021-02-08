# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the base DragTool class.
"""
# Enthought library imports
from enable.base_tool import BaseTool, KeySpec
from traits.api import Bool, Enum, List, Property, Str, Tuple, cached_property


class DragTool(BaseTool):
    """ Base class for tools that are activated by a drag operation.

    This tool insulates the drag operation from double clicks and the like, and
    gracefully manages the transition into and out of drag mode.
    """

    # The mouse button used for this drag operation.
    drag_button = Enum("left", "right")

    # End the drag operation if the mouse leaves the associated component?
    # NOTE: This behavior depends on "mouse_leave" events, which in general
    # are not fired when `capture_mouse` is True (default).
    end_drag_on_leave = Bool(False)

    # These keys, if pressed during drag, cause the drag operation to reset.
    cancel_keys = List(Str, ["Esc"])

    # The position of the initial mouse click that started the drag.
    # Typically, tools that move things around use this
    # position to do hit-testing to determine what object to "pick up".
    mouse_down_position = Tuple(0.0, 0.0)

    # The modifier key that must be used to activate the tool.
    modifier_key = Enum("none", "shift", "alt", "control")

    # Whether or not to capture the mouse during the drag operation. In effect,
    # this routes mouse events back to this tool for dispatching, rather than
    # allowing the event to be handled by the window. This may have effects
    # surrounding "mouse_leave" events: see note on `end_drag_on_leave` flag.
    capture_mouse = Bool(True)

    # ------------------------------------------------------------------------
    # Private traits used by DragTool
    # ------------------------------------------------------------------------

    # The possible states of this tool.
    _drag_state = Enum("nondrag", "dragging")

    # Records whether a mouse_down event has been received while in
    # "nondrag" state.  This is a safety check to prevent the tool from
    # suddenly getting mouse focus while the mouse button is down (either from
    # window_enter or programmatically) and erroneously
    # initiating a drag.
    _mouse_down_received = Bool(False)

    # private property to hold the current list of KeySpec instances of the
    # cancel keys
    _cancel_keys = Property(List(KeySpec), depends_on="cancel_keys")

    # ------------------------------------------------------------------------
    # Interface for subclasses
    # ------------------------------------------------------------------------

    def is_draggable(self, x, y):
        """ Returns whether the (x,y) position is in a region that is OK to
        drag.

        Used by the tool to determine when to start a drag.
        """
        return True

    def drag_start(self, event):
        """ Called when the drag operation starts.

        The *event* parameter is the mouse event that established the drag
        operation; its **x** and **y** attributes correspond to the current
        location of the mouse, and not to the position of the mouse when the
        initial left_down or right_down event happened.
        """
        pass

    def dragging(self, event):
        """ This method is called for every mouse_move event that the tool
        receives while the user is dragging the mouse.

        It is recommended that subclasses do most of their work in this method.
        """
        pass

    def drag_cancel(self, event):
        """ Called when the drag is cancelled.

        A drag is usually cancelled by receiving a mouse_leave event when
        end_drag_on_leave is True, or by the user pressing any of the
        **cancel_keys**.
        """
        pass

    def drag_end(self, event):
        """ Called when a mouse event causes the drag operation to end.
        """
        pass

    # ------------------------------------------------------------------------
    # Private methods for handling drag
    # ------------------------------------------------------------------------

    def _dispatch_stateful_event(self, event, suffix):
        # We intercept a lot of the basic events and re-map them if
        # necessary.  "consume" indicates whether or not we should pass
        # the event to the subclass's handlers.
        consume = False
        if suffix == self.drag_button + "_down":
            consume = self._drag_button_down(event)
        elif suffix == self.drag_button + "_up":
            consume = self._drag_button_up(event)
        elif suffix == "mouse_move":
            consume = self._drag_mouse_move(event)
        elif suffix == "mouse_leave":
            consume = self._drag_mouse_leave(event)
        elif suffix == "mouse_enter":
            consume = self._drag_mouse_enter(event)
        elif suffix == "key_pressed":
            consume = self._drag_cancel_keypressed(event)

        if not consume:
            BaseTool._dispatch_stateful_event(self, event, suffix)
        else:
            event.handled = True

    def _cancel_drag(self, event):
        self._drag_state = "nondrag"
        outcome = self.drag_cancel(event)
        self._mouse_down_received = False
        if event.window.mouse_owner == self:
            event.window.set_mouse_owner(None)
        return outcome

    def _drag_cancel_keypressed(self, event):
        if (self._drag_state != "nondrag"
                and any(map(lambda x: x.match(event), self._cancel_keys))):
            return self._cancel_drag(event)
        else:
            return False

    def _drag_mouse_move(self, event):
        state = self._drag_state
        button_down = getattr(event, self.drag_button + "_down")
        if state == "nondrag":
            if (button_down
                    and self._mouse_down_received
                    and self.is_draggable(*self.mouse_down_position)):
                self._drag_state = "dragging"
                if self.capture_mouse:
                    event.window.set_mouse_owner(
                        self,
                        transform=event.net_transform(),
                        history=event.dispatch_history,
                    )
                self.drag_start(event)
                return self._drag_mouse_move(event)
            return False
        elif state == "dragging":
            if button_down:
                return self.dragging(event)
            else:
                return self._drag_button_up(event)

        # If we don't invoke the subclass drag handler, then don't consume the
        # event.
        return False

    def _drag_button_down(self, event):
        if self._drag_state == "nondrag":
            self.mouse_down_position = (event.x, event.y)
            self._mouse_down_received = True
        return False

    def _drag_button_up(self, event):
        self._mouse_down_received = False
        state = self._drag_state
        if event.window.mouse_owner == self:
            event.window.set_mouse_owner(None)
        if state == "dragging":
            self._drag_state = "nondrag"
            return self.drag_end(event)

        # If we don't invoke the subclass drag handler, then don't consume the
        # event.
        return False

    def _drag_mouse_leave(self, event):
        if self.end_drag_on_leave and self._drag_state == "dragging":
            return self._cancel_drag(event)
        return False

    def _drag_mouse_enter(self, event):
        state = self._drag_state
        if state == "nondrag":
            pass
        elif state == "dragging":
            pass
        return False

    # ------------------------------------------------------------------------
    # Private methods for trait getter/setters
    # ------------------------------------------------------------------------

    @cached_property
    def _get__cancel_keys(self):
        return [KeySpec(key) for key in self.cancel_keys]
