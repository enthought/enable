# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

from traits.api import Any

from .base import BOTTOM, LEFT, RIGHT, TOP, bounds_to_coordinates
from .interactor import Interactor

# -----------------------------------------------------------------------------
#  'DragResizeHandler' class:
# -----------------------------------------------------------------------------


class DragResizeHandler(Interactor):

    # -------------------------------------------------------------------------
    #  Trait definitions:
    # -------------------------------------------------------------------------

    # User-supplied drag event validation function:
    drag_validate = Any

    # -------------------------------------------------------------------------
    #  Initialize the object:
    # -------------------------------------------------------------------------

    def init(self, component, bounds_rect, anchor, unconstrain, drag_event,
             start_event):
        self.component = component
        self.bounds_rect = bounds_rect
        self.anchor = anchor
        self.unconstrain = unconstrain
        self.drag_event = drag_event
        self.start_event = start_event
        self.drag_x = self.start_x = start_event.x
        self.drag_y = self.start_y = start_event.y

        # Get the coordinates of the anchor point:
        xl, yb, xr, yt = bounds_to_coordinates(component.bounds)
        if (anchor & LEFT) != 0:
            self.anchor_x = xl
            self.float_x = xr
        else:
            self.anchor_x = xr
            self.float_x = xl
        if (anchor & BOTTOM) != 0:
            self.anchor_y = yb
            self.float_y = yt
        else:
            self.anchor_y = yt
            self.float_y = yb

        # Set up the drag termination handler:
        self.on_trait_change(self.drag_done, drag_event)

    # -------------------------------------------------------------------------
    #  Handle the mouse moving while resizing:
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

        # Calculate the new 'floating' point:
        unconstrain = self.unconstrain
        ax, ay = self.anchor_x, self.anchor_y
        nx, ny = self.float_x, self.float_y
        if (unconstrain & (LEFT | RIGHT)) != 0:
            nx = self.float_x + x - self.start_x
            if nx > ax:
                if (unconstrain & RIGHT) == 0:
                    nx = ax
            elif nx < ax:
                if (unconstrain & LEFT) == 0:
                    nx = ax
        if (unconstrain & (TOP | BOTTOM)) != 0:
            ny = self.float_y + y - self.start_y
            if ny > ay:
                if (unconstrain & TOP) == 0:
                    ny = ay
            elif ny < ay:
                if (unconstrain & BOTTOM) == 0:
                    ny = ay

        # Make sure the new point is inside the drag bounds (if required):
        if self.bounds_rect is not None:
            bxl, byb, bxr, byt = self.bounds_rect
            nx = max(min(nx, bxr), bxl)
            ny = max(min(ny, byt), byb)

        # Calculate the new size of the component and make sure that it meets
        # the min and max size requirements for the component:
        component = self.component
        mindx, maxdx = component.min_width, component.max_width
        mindy, maxdy = component.min_height, component.max_height
        ndx, ndy = abs(nx - ax) + 1, abs(ny - ay) + 1
        if ndx > maxdx:
            if nx > ax:
                nx = ax + maxdx
            else:
                nx = ax - maxdx
        elif ndx < mindx:
            if nx < ax:
                nx = ax - mindx
            elif (nx > ax) or ((unconstrain & RIGHT) != 0):
                nx = ax + mindx
            else:
                nx = ax - mindx
        if ndy > maxdy:
            if ny > ay:
                ny = ay + maxdy
            else:
                ny = ay - maxdy
        elif ndy < mindy:
            if ny < ay:
                ny = ay - mindy
            elif (ny > ay) or ((unconstrain & TOP) != 0):
                ny = ay + mindy
            else:
                ny = ay - mindy

        # Update the bounds of the component:
        bounds = (min(nx, ax), min(ny, ay), abs(nx - ax) + 1, abs(ny - ay) + 1)
        if self.drag_validate is not None:
            bounds = self.drag_validate(event, bounds)
        if bounds != component.bounds:
            component.bounds = bounds

            # Tell the 'paint' routine we are doing a drag resize operation:
            event.window.drag_resize_update()

    # -------------------------------------------------------------------------
    #  Handle the user releasing the original drag button:
    # -------------------------------------------------------------------------

    def drag_done(self, event):
        # 'Unhook' the drag done notification handler:
        self.on_trait_change(self.drag_done, self.drag_event, remove=True)

        # Inform the component that the resize operation is complete:
        self.component.resized = True
