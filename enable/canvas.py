# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the enable Canvas class """

# Enthought library imports
from traits.api import Bool, List, Trait, Tuple
from kiva.api import FILL


# Local relative imports
from .component import Component
from .container import Container


class Canvas(Container):
    """
    An infinite canvas with components on it.  It can optionally be given
    a "view region" which will be used as the notional bounds of the
    canvas in all operations that require bounds.

    A Canvas can be nested inside another container, but usually a
    viewport is more appropriate.

    Note: A Canvas has infinite bounds, but its .bounds attribute is
    overloaded to be something more meaningful, namely, the bounding
    box of its child components and the optional view area of the
    viewport that is looking at it.  (TODO: add support for multiple
    viewports.)
    """

    # This optional tuple of (x,y,x2,y2) allows viewports to inform the canvas
    # of the "region of interest" that it should use when computing its
    # notional bounds for clipping and event handling purposes.  If this trait
    # is None, then the canvas really does behave as if it has no bounds.
    view_bounds = Trait(None, None, Tuple)

    # The (x,y) position of the lower-left corner of the rectangle
    # corresponding to the dimensions in self.bounds.  Unlike self.position,
    # this position is in the canvas's space, and not in the coordinate space
    # of the parent.
    bounds_offset = List

    draw_axes = Bool(False)

    # ------------------------------------------------------------------------
    # Inherited traits
    # ------------------------------------------------------------------------

    # Use the auto-size/fit_components mechanism to ensure that the bounding
    # box around our inner components gets updated properly.
    auto_size = True
    fit_components = "hv"

    # The following traits are ignored, but we set them to sensible values.
    fit_window = False
    resizable = "hv"

    # ------------------------------------------------------------------------
    # Protected traits
    # ------------------------------------------------------------------------

    # The (x, y, x2, y2) coordinates of the bounding box of the components
    # in our inner coordinate space
    _bounding_box = Tuple((0, 0, 100, 100))

    def compact(self):
        """
        Wraps the superclass method to also take into account the view
        bounds (if they are present
        """
        self._bounding_box = self._calc_bounding_box()
        self._view_bounds_changed()

    def is_in(self, x, y):
        return True

    def remove(self, *components):
        """ Removes components from this container """
        needs_compact = False
        for component in components:
            if component in self._components:
                component.container = None
                self._components.remove(component)
            else:
                raise RuntimeError(
                    "Unable to remove component from container."
                )

            # Check to see if we need to compact.
            x, y, x2, y2 = self._bounding_box
            if ((component.outer_x2 == x2 - x)
                    or (component.outer_y2 == y2 - y)
                    or (component.x == 0)
                    or (component.y == 0)):
                needs_compact = True

        if needs_compact:
            self.compact()
        self.invalidate_draw()

    def draw(self, gc, view_bounds=None, mode="normal"):
        if self.view_bounds is None:
            self.view_bounds = view_bounds
        super(Canvas, self).draw(gc, view_bounds, mode)

    # ------------------------------------------------------------------------
    # Protected methods
    # ------------------------------------------------------------------------

    def _should_compact(self):
        if self.auto_size:
            if self.view_bounds is not None:
                llx, lly = self.view_bounds[:2]
            else:
                llx = lly = 0
            for component in self.components:
                if (component.outer_x2 >= self.width
                        or component.outer_y2 >= self.height
                        or component.outer_x < llx
                        or component.outer_y < lly):
                    return True
        else:
            return False

    def _draw_background(self, gc, view_bounds=None, mode="default"):
        if self.bgcolor not in ("clear", "transparent", "none"):
            if self.view_bounds is not None:
                x, y, x2, y2 = self.view_bounds
            else:
                x, y, x2, y2 = self._bounding_box
            r = (x, y, x2 - x + 1, y2 - y + 1)

            with gc:
                gc.set_antialias(False)
                gc.set_fill_color(self.bgcolor_)
                gc.draw_rect(r, FILL)

        # Call the enable _draw_border routine
        if not self.overlay_border and self.border_visible:
            # Tell _draw_border to ignore the self.overlay_border
            self._draw_border(gc, view_bounds, mode, force_draw=True)

    def _draw_underlay(self, gc, view_bounds=None, mode="default"):
        if self.draw_axes:
            x, y, x2, y2 = self.view_bounds
            if (x <= 0 <= x2) or (y <= 0 <= y2):
                with gc:
                    gc.set_stroke_color((0, 0, 0, 1))
                    gc.set_line_width(1.0)
                    gc.move_to(0, y)
                    gc.line_to(0, y2)
                    gc.move_to(x, 0)
                    gc.line_to(x2, 0)
                    gc.stroke_path()
        super(Container, self)._draw_underlay(gc, view_bounds, mode)

    def _transform_view_bounds(self, view_bounds):
        # Overload the parent class's implementation to skip visibility test
        if view_bounds:
            v = view_bounds
            new_bounds = (v[0] - self.x, v[1] - self.y, v[2], v[3])
        else:
            new_bounds = None
        return new_bounds

    # ------------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------------

    def _bounds_offset_default(self):
        return [0, 0]

    def _view_bounds_changed(self):
        llx, lly, urx, ury = self._bounding_box
        if self.view_bounds is not None:
            x, y, x2, y2 = self.view_bounds
            llx = min(llx, x)
            lly = min(lly, y)
            urx = max(urx, x2)
            ury = max(ury, y2)
        self.bounds_offset = [llx, lly]
        self.bounds = [urx - llx + 1, ury - lly + 1]

    # Override Container.bounds_changed so that _layout_needed is not
    # set.  Containers need to invalidate layout because they act as
    # sizers, but the Canvas is unbounded and thus does not need to
    # invalidate layout.
    def _bounds_changed(self, old, new):
        Component._bounds_changed(self, old, new)
        self.invalidate_draw()

    def _bounds_items_changed(self, event):
        Component._bounds_items_changed(self, event)
        self.invalidate_draw()
