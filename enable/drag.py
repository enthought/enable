# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

from .base import (
    add_rectangles, bounds_to_coordinates, intersect_coordinates, send_event_to
)
from .events import DragEvent
from .interactor import Interactor

# ---------------------------------------------------------------------------
#  'DragHandler' class:
# ---------------------------------------------------------------------------


class DragHandler(Interactor):

    # -------------------------------------------------------------------------
    #  Initialize the object:
    # -------------------------------------------------------------------------

    def init(self, components, bounds_rect, drag_root, drag_bounds_rect,
             drag_copy, drag_event, start_event):
        self.components = components[:]
        self.bounds_rect = bounds_rect
        self.drag_root = drag_root
        self.drag_bounds_rect = self.drag_bounds_rect_start = drag_bounds_rect
        self.drag_copy = drag_copy
        self.drag_event = drag_event
        self.start_event = start_event
        self.drag_x = self.start_x = start_event.x
        self.drag_y = self.start_y = start_event.y
        self.window = components[0].window
        self.drag_over = []

        self.on_trait_change(self.drag_done, drag_event)

    # -------------------------------------------------------------------------
    #  Handle the mouse moving while dragging:
    # -------------------------------------------------------------------------

    def _mouse_move_changed(self, event):
        # Get the current drag location:
        x = event.x
        y = event.y

        # If the mouse did not actually move, then ignore the event:
        if (x == self.drag_x) and (y == self.drag_y):
            return

        # Save the new position:
        self.drag_x = x
        self.drag_y = y

        # Calculate the distance moved from the original starting point:
        dx = x - self.start_x
        dy = y - self.start_y

        # Calculate the new drag bounds:
        xl, yb, xr, yt = add_rectangles(
            self.drag_bounds_rect_start, (dx, dy, dx, dy)
        )

        # Adjust the new drag location if it is not completely within the drag
        # bounds. Exit if the result is the same as the previous location:
        if self.bounds_rect is not None:
            bxl, byb, bxr, byt = self.bounds_rect
            if xl < bxl:
                xr += bxl - xl
                xl = bxl
            elif xr > bxr:
                xl -= xr - bxr
                xr = bxr
            if yb < byb:
                yt += byb - yb
                yb = byb
            elif yt > byt:
                yb -= yt - byt
                yt = byt
            x = xl - self.drag_bounds_rect[0] + self.drag_x
            y = yb - self.drag_bounds_rect[1] + self.drag_y

        # If the new drag bounds is the same as the last one, nothing changed:
        drag_bounds_rect = bounds_to_coordinates(
            self.drag_validate(event, (xl, yb, xr - xl, yt - yb))
        )
        if drag_bounds_rect == self.drag_bounds_rect:
            return

        # Update the drag bounds and current drag location:
        self.drag_bounds_rect = drag_bounds_rect

        # Notify each drag component of its new drag location:
        dx = drag_bounds_rect[0] - self.drag_bounds_rect_start[0]
        dy = drag_bounds_rect[1] - self.drag_bounds_rect_start[1]
        for component in self.components:
            cx, cy = component.location()
            component.dragged = DragEvent(
                x=cx + dx,
                y=cy + dy,
                x0=self.start_x,
                y0=self.start_y,
                copy=self.drag_copy,
                components=self.components,
                start_event=self.start_event,
                window=event.window,
            )

        # Process the 'drag_over' events for any objects being dragged over:
        drag_over_event = DragEvent(
            x=x,
            y=y,
            x0=self.start_x,
            y0=self.start_y,
            copy=self.drag_copy,
            components=self.components,
            start_event=self.start_event,
            window=event.window,
        )
        new_drag_over = []
        cur_drag_over = self.drag_over
        for component in self.drag_root.components_at(x, y):
            new_drag_over.append(component)
            if component in cur_drag_over:
                cur_drag_over.remove(component)
                component.drag_over = drag_over_event
            else:
                component.drag_enter = drag_over_event
        for component in cur_drag_over:
            component.drag_leave = drag_over_event
        self.drag_over = new_drag_over

        # Tell the Enable window where the drag is now:
        self.window.set_drag_bounds_rect(drag_bounds_rect)

    # -------------------------------------------------------------------------
    #  Validate the proposed new drag location (by default, just accept it):
    # -------------------------------------------------------------------------

    def drag_validate(self, event, drag_bounds):
        return drag_bounds

    # -------------------------------------------------------------------------
    #  Handle the user releasing the original drag button:
    # -------------------------------------------------------------------------

    def drag_done(self, event):
        components = self.components
        drag_copy = self.drag_copy
        start_event = self.start_event

        # 'Unhook' the drag done notification handler:
        self.on_trait_change(self.drag_done, self.drag_event, remove=True)

        # Compute the new drag bounds:
        x = event.x
        y = event.y
        dx = x - self.start_x
        dy = y - self.start_y
        xl, yb, xr, yt = add_rectangles(
            self.drag_bounds_rect_start, (dx, dy, dx, dy)
        )
        drag_bounds_rect = bounds_to_coordinates(
            self.drag_validate(event, (xl, yb, xr - xl, yt - yb))
        )

        # If the new bounds are not within the drag area, use the last drag
        # location:
        if (self.bounds_rect is not None
            and (intersect_coordinates(self.bounds_rect, drag_bounds_rect)
                 != drag_bounds_rect)):
            drag_bounds_rect = self.drag_bounds_rect

        # Notify each dragged component where it was dropped:
        dx = drag_bounds_rect[0] - self.drag_bounds_rect_start[0]
        dy = drag_bounds_rect[1] - self.drag_bounds_rect_start[1]
        for component in components:
            cx, cy = component.location()
            component.dropped = DragEvent(
                x=cx + dx,
                y=cy + dy,
                x0=self.start_x,
                y0=self.start_y,
                copy=drag_copy,
                components=components,
                start_event=start_event,
                window=event.window,
            )

        # Process the 'dropped_on' event for the object(s) it was dropped on:
        components_at = self.drag_root.components_at(x, y)
        drag_event = DragEvent(
            x=self.start_x + dx,
            y=self.start_y + dy,
            x0=self.start_x,
            y0=self.start_y,
            copy=drag_copy,
            components=components,
            start_event=start_event,
            window=event.window,
        )
        index = send_event_to(components_at, "dropped_on", drag_event)

        # Send all the runner-ups a 'drag_leave' consolation prize:
        drag_over = self.drag_over
        for component in components_at[0:index]:
            if component in drag_over:
                component.drag_leave = drag_event

        # Make sure all of the dragged components are drawable again:
        if not drag_copy:
            for component in components:
                component._drawable = True

        # Redraw the window to clean-up any drag artifacts:
        self.window.redraw()

        # Finally, release any references we have been using:
        self.components[:] = []
        self.drag_over = self.drag_root = self.window = None
