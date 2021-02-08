# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Cross-hair tool for measuring SVG rendering results.
"""

from enable.api import BaseTool, ColorTrait, LineStyle
from traits.api import Bool, Float, HasTraits, List, Tuple, on_trait_change


class Crosshair(BaseTool):
    """ Display a crosshair at the given SVG coordinates.

    This will do the appropriate transformations in order to map Enable
    coordinates to SVG coordinates.
    """

    svg_coords = Tuple(Float, Float)
    line_color = ColorTrait("black")
    line_width = Float(1.0)
    line_style = LineStyle("solid")

    # Whether the mouse is currently inside the component or not.
    mouse_in = Bool(False)

    visible = True
    draw_mode = "overlay"

    def draw(self, gc, view_bounds=None):
        """ Draws this tool on a graphics context.

        It is assumed that the graphics context has a coordinate transform that
        matches the origin of its component. (For containers, this is just the
        origin; for components, it is the origin of their containers.)
        """

        if not self.mouse_in:
            return
        # Convert from SVG coordinates to Enable coordinates.
        h = self.component.height
        x, y0 = self.svg_coords
        y = h - y0
        gc.save_state()
        try:
            gc.set_stroke_color(self.line_color_)
            gc.set_line_width(self.line_width)
            gc.set_line_dash(self.line_style_)
            gc.move_to(self.component.x, y + 0.5)
            gc.line_to(self.component.x2, y + 0.5)
            gc.move_to(x - 0.5, self.component.y)
            gc.line_to(x - 0.5, self.component.y2)
            gc.stroke_path()
        finally:
            gc.restore_state()

    def overlay(self, component, gc, view_bounds=None, mode="normal"):
        """ Draws this component overlaid on a graphics context.
        """
        self.draw(gc, view_bounds)

    def do_layout(self, *args, **kw):
        pass

    def normal_mouse_enter(self, event):
        self.mouse_in = True

    def normal_mouse_leave(self, event):
        self.mouse_in = False

    def normal_mouse_move(self, event):
        """ Handles the mouse being moved.
        """
        if self.component is None:
            return
        # Map the Enable coordinates of the event to SVG coordinates.
        h = self.component.height
        y = h - event.y
        self.svg_coords = event.x, y
        event.handled = True

    @on_trait_change("svg_coords,mouse_in")
    def ensure_redraw(self):
        if self.component is not None:
            self.component.invalidate_and_redraw()


class MultiController(HasTraits):
    """ Keep multiple Crosshairs in sync.
    """

    svg_coords = Tuple(Float, Float)
    mouse_in = Bool(False)
    crosshairs = List()

    def __init__(self, *crosshairs, **traits):
        super(MultiController, self).__init__(**traits)
        for ch in crosshairs:
            self.add(ch)

    def add(self, crosshair):
        """ Synch a new Crosshair.
        """
        if crosshair not in self.crosshairs:
            self.sync_trait("svg_coords", crosshair)
            self.sync_trait("mouse_in", crosshair)
            self.crosshairs.append(crosshair)

    def remove(self, crosshair):
        """ Unsynch a recorded Crosshair.
        """
        if crosshair in self.crosshairs:
            self.sync_trait("svg_coords", crosshair, remove=True)
            self.sync_trait("mouse_in", crosshair, remove=True)
            self.crosshairs.append(crosshair)
