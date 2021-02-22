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
Value Drag Tools
================

This module contains tools to handle simple dragging interactions where the
drag operation has a direct effect on some underlying value.  This can
potentially be used as the basis for many different interactions,
"""
from traits.api import (
    Any, Bool, Dict, Either, Enum, Event, Float, Set, Str, Tuple
)
from .drag_tool import DragTool

keys = {"shift", "alt", "control"}


class IdentityMapper(object):
    def map_data(self, screen):
        return screen

    def map_screen(self, data):
        return data


identity_mapper = IdentityMapper()


class ValueDragTool(DragTool):
    """ Abstract tool for modifying a value as the mouse is dragged

    The tool allows the use of an x_mapper and y_mapper to map between
    screen coordinates and more abstract data coordinates.  These mappers must
    be objects with a map_data() method that maps a component-space coordinate
    to a data-space coordinate.  Chaco mappers satisfy the required API, and
    the tool will look for 'x_mapper' and 'y_mapper' attributes on the
    component to use as the defaults, facilitating interoperability with Chaco
    plots. Failing that, a simple identity mapper is provided which does
    nothing. Coordinates are given relative to the component.

    Subclasses of this tool need to supply get_value() and set_delta() methods.
    The get_value method returns the current underlying value, while the
    set_delta method takes the current mapped x and y deltas from the original
    position, and sets the underlying value appropriately.  The object stores
    the original value at the start of the operation as the original_value
    attribute.
    """

    #: set of modifier keys that must be down to invoke the tool
    modifier_keys = Set(Enum(*keys))

    #: mapper that maps from horizontal screen coordinate to data coordinate
    x_mapper = Any

    #: mapper that maps from vertical screen coordinate to data coordinate
    y_mapper = Any

    #: start point of the drag in component coordinates
    original_screen_point = Tuple(Float, Float)

    #: start point of the drag in data coordinates
    original_data_point = Tuple(Any, Any)

    #: initial underlying value
    original_value = Any

    #: new_value event for inspector overlay
    new_value = Event(Dict)

    #: visibility for inspector overlay
    visible = Bool(False)

    def get_value(self):
        """ Return the current value that is being modified
        """
        pass

    def set_delta(self, value, delta_x, delta_y):
        """ Set the value that is being modified

        This function should modify the underlying value based on the provided
        delta_x and delta_y in data coordinates.  These deltas are total
        displacement from the original location, not incremental.  The value
        parameter is the original value at the point where the drag started.
        """
        pass

    # Drag tool API

    def drag_start(self, event):
        self.original_screen_point = (event.x, event.y)
        data_x = self.x_mapper.map_data(event.x)
        data_y = self.y_mapper.map_data(event.y)
        self.original_data_point = (data_x, data_y)
        self.original_value = self.get_value()
        self.visible = True
        return True

    def dragging(self, event):
        position = event.current_pointer_position()
        delta_x = (
            self.x_mapper.map_data(position[0]) - self.original_data_point[0]
        )
        delta_y = (
            self.y_mapper.map_data(position[1]) - self.original_data_point[1]
        )
        self.set_delta(self.original_value, delta_x, delta_y)
        return True

    def drag_end(self, event):
        event.window.set_pointer("arrow")
        self.visible = False
        return True

    def _drag_button_down(self, event):
        # override button down to handle modifier keys correctly
        if not event.handled and self._drag_state == "nondrag":
            key_states = dict((key, key in self.modifier_keys) for key in keys)
            if (not all(getattr(event, key + "_down") == state
                        for key, state in key_states.items())):
                return False
            self.mouse_down_position = (event.x, event.y)
            if not self.is_draggable(*self.mouse_down_position):
                self._mouse_down_recieved = False
                return False
            self._mouse_down_received = True
            return True
        return False

    # traits default handlers

    def _x_mapper_default(self):
        # if the component has an x_mapper, try to use it by default
        return getattr(self.component, "x_mapper", identity_mapper)

    def _y_mapper_default(self):
        # if the component has an x_mapper, try to use it by default
        return getattr(self.component, "y_mapper", identity_mapper)


class AttributeDragTool(ValueDragTool):
    """ Tool which modifies a model's attributes as it drags

    This is designed to cover the simplest of drag cases where the drag is
    modifying one or two numerical attributes on an underlying model.  To use,
    simply provide the model object and the attributes that you want to be
    changed by the drag.  If only one attribute is required, the other can be
    left as an empty string.
    """

    #: the model object which has the attributes we are modifying
    model = Any

    #: the name of the attributes that is modified by horizontal motion
    x_attr = Str

    #: the name of the attributes that is modified by vertical motion
    y_attr = Str

    #: max and min values for x value
    x_bounds = Tuple(Either(Float, Str, None), Either(Float, Str, None))

    #: max and min values for y value
    y_bounds = Tuple(Either(Float, Str, None), Either(Float, Str, None))

    x_name = Str

    y_name = Str

    # ValueDragTool API

    def get_value(self):
        """ Get the current value of the attributes

        Returns a 2-tuple of (x, y) values.  If either x_attr or y_attr is
        the empty string, then the corresponding component of the tuple is
        None.
        """
        x_value = None
        y_value = None
        if self.x_attr:
            x_value = getattr(self.model, self.x_attr)
        if self.y_attr:
            y_value = getattr(self.model, self.y_attr)
        return (x_value, y_value)

    def set_delta(self, value, delta_x, delta_y):
        """ Set the current value of the attributes

        Set the underlying attribute values based upon the starting value and
        the provided deltas.  The values are simply set to the sum of the
        appropriate coordinate and the delta. If either x_attr or y_attr is
        the empty string, then the corresponding component of is ignored.

        Note that setting x and y are two separate operations, and so will fire
        two trait notification events.
        """
        inspector_value = {}
        if self.x_attr:
            x_value = value[0] + delta_x
            if self.x_bounds[0] is not None:
                if isinstance(self.x_bounds[0], str):
                    m = getattr(self.model, self.x_bounds[0])
                else:
                    m = self.x_bounds[0]
                x_value = max(x_value, m)
            if self.x_bounds[1] is not None:
                if isinstance(self.x_bounds[1], str):
                    M = getattr(self.model, self.x_bounds[1])
                else:
                    M = self.x_bounds[1]
                x_value = min(x_value, M)
            setattr(self.model, self.x_attr, x_value)
            inspector_value[self.x_name] = x_value
        if self.y_attr:
            y_value = value[1] + delta_y
            if self.y_bounds[0] is not None:
                if isinstance(self.y_bounds[0], str):
                    m = getattr(self.model, self.y_bounds[0])
                else:
                    m = self.y_bounds[0]
                y_value = max(y_value, m)
            if self.y_bounds[1] is not None:
                if isinstance(self.y_bounds[1], str):
                    M = getattr(self.model, self.y_bounds[1])
                else:
                    M = self.y_bounds[1]
                y_value = min(y_value, M)
            setattr(self.model, self.y_attr, y_value)
            inspector_value[self.y_name] = y_value
        self.new_value = inspector_value

    def _x_name_default(self):
        return self.x_attr.replace("_", " ").capitalize()

    def _y_name_default(self):
        return self.y_attr.replace("_", " ").capitalize()
