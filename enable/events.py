# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""
Define the event objects and traits used by Enable components.

For a list of all the possible event suffixes, see interactor.py.
"""
from functools import reduce

# Major library imports
from numpy import array, dot

# Enthought imports
from kiva import affine
from traits.api import Any, Bool, Float, HasTraits, Int, Event, List, ReadOnly


class BasicEvent(HasTraits):

    x = Float
    y = Float

    # True if the event has been handled.
    handled = Bool(False)

    # The AbstractWindow instance through/from which this event was fired.
    # Can be None.
    window = Any

    # (x,y) position stack; initialized to an empty list
    _pos_stack = List(())

    # Affine transform stack; initialized to an empty list
    _transform_stack = List(())

    # This is a list of objects that have transformed the event's
    # coordinates.  This can be used to recreate the dispatch path
    # that the event took.
    dispatch_history = List()

    def push_transform(self, transform, caller=None):
        """
        Saves the current transform in a stack and sets the given transform
        to be the active one.
        """
        x, y = dot(array((self.x, self.y, 1)), transform)[:2]
        self._pos_stack.append((self.x, self.y))
        self._transform_stack.append(transform)
        self.x = x
        self.y = y
        if caller is not None:
            self.dispatch_history.append(caller)

    def pop(self, count=1, caller=None):
        """
        Restores a previous position of the event.  If **count** is provided,
        then pops **count** elements off of the event stack.
        """
        for i in range(count - 1):
            self._pos_stack.pop()
            self._transform_stack.pop()
        self.x, self.y = self._pos_stack.pop()
        self._transform_stack.pop()
        if caller is not None:
            if caller == self.dispatch_history[-1]:
                self.dispatch_history.pop()

    def offset_xy(self, origin_x, origin_y, caller=None):
        r"""
        Shifts this event to be in the coordinate frame whose origin, specified
        in the event's coordinate frame, is (origin_x, origin_y).

        Basically, a component calls event.offset_xy(\*self.position) to shift
        the event into its own coordinate frame.
        """
        self.push_transform(
            affine.affine_from_translation(-origin_x, -origin_y)
        )
        if caller is not None:
            self.dispatch_history.append(caller)

    def scale_xy(self, scale_x, scale_y, caller=None):
        """
        Scales the event to be in the scale specified.

        A component calls event.scale_xy(scale) to scale the event into its own
        coordinate frame when the ctm has been scaled.  This operation is used
        for zooming.
        """
        # Note that the meaning of scale_x and scale_y for Enable
        # is the inverted from the meaning for Kiva.affine.
        # TODO: Fix this discrepancy.
        self.push_transform(affine.affine_from_scale(1 / scale_x, 1 / scale_y))
        if caller is not None:
            self.dispatch_history.append(caller)

    def net_transform(self):
        """
        Returns a single transformation (currently only (dx,dy)) that reflects
        the total amount of change from the original coordinates to the current
        offset coordinates stored in self.x and self.y.
        """
        if len(self._transform_stack) == 0:
            return affine.affine_identity()
        else:
            return reduce(dot, self._transform_stack[::-1])

    def current_pointer_position(self):
        """
        Returns the current pointer position in the transformed coordinates
        """
        window_pos = self.window.get_pointer_position()
        return tuple(dot(array(window_pos + (1,)), self.net_transform())[:2])

    def __repr__(self):
        s = "%s(x=%r, y=%r, handled=%r)" % (
            self.__class__.__name__,
            self.x,
            self.y,
            self.handled,
        )
        return s


class MouseEvent(BasicEvent):
    alt_down = ReadOnly
    control_down = ReadOnly
    shift_down = ReadOnly
    left_down = ReadOnly
    middle_down = ReadOnly
    right_down = ReadOnly
    mouse_wheel = ReadOnly
    mouse_wheel_axis = ReadOnly
    mouse_wheel_delta = ReadOnly


mouse_event_trait = Event(MouseEvent)


class DragEvent(BasicEvent):
    """ A system UI drag-and-drop operation.  This is not the same as a
    DragTool event.
    """

    x0 = Float
    y0 = Float
    copy = ReadOnly
    obj = ReadOnly
    start_event = ReadOnly

    def __repr__(self):
        s = "%s(x=%r, y=%r, x0=%r, y0=%r, handled=%r)" % (
            self.__class__.__name__,
            self.x,
            self.y,
            self.x0,
            self.y0,
            self.handled,
        )
        return s


drag_event_trait = Event(DragEvent)


class KeyEvent(BasicEvent):
    event_type = (
        ReadOnly
    )  # one of 'key_pressed', 'key_released' or 'character'

    # 'character' is a single unicode character or is a string describing the
    # high-bit and control characters.  (See module enable.toolkit_constants)
    # depending on the event type, it may represent the physical key pressed,
    # or the text that was generated by a keystroke
    character = ReadOnly

    alt_down = ReadOnly
    control_down = ReadOnly
    shift_down = ReadOnly

    event = ReadOnly  # XXX the underlying toolkit's event object, remove?

    def __repr__(self):
        s = (
            ("%s(event_type=%r, character=%r, alt_down=%r, control_down=%r, "
             "shift_down=%r, handled=%r)")
            % (
                self.__class__.__name__,
                self.event_type,
                self.character,
                self.alt_down,
                self.control_down,
                self.shift_down,
                self.handled,
            )
        )
        return s


key_event_trait = Event(KeyEvent)


class BlobEvent(BasicEvent):
    """ Represent a single pointer event from a multi-pointer event system.

    Will be used with events:
        blob_down
        blob_move
        blob_up
    """

    # The ID of the pointer.
    bid = Int(-1)

    # If a blob_move event, then these will be the coordinates of the blob at
    # the previous frame.
    x0 = Float(0.0)
    y0 = Float(0.0)

    def push_transform(self, transform, caller=None):
        """ Saves the current transform in a stack and sets the given transform
        to be the active one.

        This will also adjust x0 and y0.
        """
        x, y = dot(array((self.x, self.y, 1)), transform)[:2]
        self._pos_stack.append((self.x, self.y))
        self._transform_stack.append(transform)
        self.x = x
        self.y = y
        x0, y0 = dot(array((self.x0, self.y0, 1)), transform)[:2]
        self.x0 = x0
        self.y0 = y0
        if caller is not None:
            self.dispatch_history.append(caller)

    def __repr__(self):
        s = "%s(bid=%r, x=%r, y=%r, x0=%r, y0=%r, handled=%r)" % (
            self.__class__.__name__,
            self.bid,
            self.x,
            self.y,
            self.x0,
            self.y0,
            self.handled,
        )
        return s


blob_event_trait = Event(BlobEvent)


class BlobFrameEvent(BasicEvent):
    """ Represent the framing events for a multi-pointer event system.

    Will be used with events:
        blob_frame_begin
        blob_frame_end

    These can be used to synchronize the effects of multiple pointers.

    The position traits are meaningless. These events will get passed down
    through all components. Also, no component should mark it as handled. The
    event must be dispatched through whether the component takes action based
    on it or not.

    NOTE: Frames without any blob events may or may not generate
    BlobFrameEvents.
    """

    # The ID number of the frame. This is generally implemented as a counter.
    # Adjacent frames should have different frame IDs, but it is permitted for
    # the counter to wrap around eventually or for the Enable application to
    # disconnect and reconnect to a multi-pointer system and have the counter
    # reset to 0.
    fid = Int(-1)

    # The timestamp of the frame in seconds from an unspecified origin.
    t = Float(0.0)

    # Never mark this event as handled. Let every component respond to it.
    # handled = ReadOnly(False)

    def __repr__(self):
        s = "%s(fid=%r, t=%r)" % (self.__class__.__name__, self.fid, self.t)
        return s


blob_frame_event_trait = Event(BlobFrameEvent)
