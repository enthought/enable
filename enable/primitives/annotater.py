"""
Define an Annotater component that allows a user to annotate an underlying
component
"""

from __future__ import with_statement

from traits.api import Event, Trait, TraitPrefixList
from traitsui.api import Group, View

from enable.api import Component
from enable.colors import ColorTrait


class Annotater(Component):

    color = ColorTrait((0.0, 0.0, 0.0, 0.2 ))
    style = Trait("rectangular", TraitPrefixList(["rectangular", 'freehand']))
    annotation = Event

    traits_view = View(Group('<component>', id = 'component'),
                       Group('<links>', id = 'links'),
                       Group('color', 'style', id = 'annotater',
                             style = 'custom'))


    #---------------------------------------------------------------------------
    # Mouse event handlers
    #---------------------------------------------------------------------------

    def _left_down_changed(self, event):
        event.handled = True
        self.window.mouse_owner = self
        self._cur_x, self._cur_y = event.x, event.y
        self._start_x, self._start_y = event.x, event.y
        return

    def _left_up_changed(self, event):
        event.handled = True
        self.window.mouse_owner = None
        if self.xy_in_bounds(event):
            self.annotation = (min(self._start_x, event.x),
                               min(self._start_y, event.y),
                               abs(self._start_x - event.x),
                               abs(self._start_y - event.y))
        self._start_x = self._start_y = self._cur_x = self._cur_y = None
        self.redraw()
        return

    def _mouse_move_changed(self, event):
        event.handled = True
        if self._start_x is not None:
            x = max(min(event.x, self.right - 1.0), self.x)
            y = max(min(event.y, self.top - 1.0), self.y)
            if (x != self._cur_x) or (y != self._cur_y):
                self._cur_x, self._cur_y = x, y
                self.redraw()
        return

    #---------------------------------------------------------------------------
    # "Component" interface
    #---------------------------------------------------------------------------

    def _draw (self, gc):
        "Draw the contents of the control"
        if self._start_x is not None:
            with gc:
                gc.set_fill_color(self.color_)
                gc.begin_path()
                gc.rect(min(self._start_x, self._cur_x),
                         min(self._start_y, self._cur_y),
                         abs(self._start_x - self._cur_x),
                         abs(self._start_y - self._cur_y))
                gc.fill_path()
        return

# EOF
