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
This is a larger example which shows how to build an editor for gradients
in a fairly standard way.

The base is the GradientBox class which handles the rendering of the gradient,
and mapping back and forth between the display and the underlying gradient
object.  It also holds a particular selected color stop in the gradient.

An overlay provides rendering of the location of the color stops in the
gradient, including a highlight for the selected color stop.

A series of tools handle interactions, including selection, dragging, adding
and deleting color stops from the underlying gradient.
"""
import numpy as np

from traits.api import Instance, Int, observe
from traitsui.api import Item, OKCancelButtons, View

from enable.example_support import DemoFrame, demo_main
from enable.api import (
    AbstractOverlay, BaseTool, ColorStop, Component, Gradient,
    LinearGradientBrush, black_color_trait, marker_trait
)
from enable.tools.api import AttributeDragTool


class GradientBox(Component):
    """ A component which draws a gradient.

    This class also provides helper methods for mapping between screen space
    and offsets, testing for hits against the location of stops, and adding
    new color stops.  These can be used by tools an overlays to render
    additional features.
    """

    #: The gradient being viewed.
    gradient = Instance(Gradient, args=(), allow_none=False)

    #: The currently selected color stop, or None if nothing is selected.
    selected = Instance(ColorStop, update=True)

    #: A linear brush that renders the gradient from left to right.
    _brush = Instance(LinearGradientBrush, update=True)

    def map_screen(self, offset):
        """ Map an offset between 0 and 1 to a screen x-value.

        Parameters
        ----------
        offset : float or array between 0.0 and 1.0
            A float or array of floats holding the offset value(s).

        Returns
        -------
        x : float or array
            The x-coordinate, or an array of x-coordinates.
        """
        return offset * self.width + self.padding_left

    def map_data(self, x):
        """ Map an x- value to an offset between 0 and 1.

        Parameters
        ----------
        x : float or array
            The x-coordinate, or an array of x-coordinates.

        Returns
        -------
        offset : float or array between 0.0 and 1.0
            A float or array of floats holding the offset value(s).
        """
        return min(1.0, max(0.0, (x - self.padding_left) / self.width))

    def hittest(self, x, tolerance=4):
        """ Return the closest stop, or None if not withing tolerance.

        Parameters
        ----------
        x : float
            The x-coordinate, or an array of x-coordinates.
        tolerance : float
            The distance in pixiels in the x direction from the actual
            location that can still be considered to be hitting the
            location of a color stop.

        Returns
        -------
        stop : ColorStop or None
            The closest color stop to the provided x-coordinate, or None if
            the closest stop is more than tolerance pixels away.
        """
        gradient = self.gradient
        positions = self.map_screen(np.array([
            stop.offset for stop in gradient.stops
        ]))

        deltas = np.abs(positions - x)
        closest = deltas.argmin()

        if deltas[closest] <= tolerance:
            return gradient.stops[closest]
        else:
            return None

    def add_stop_at(self, offset, color=None):
        """ Add a color stop to the gradient.

        Parameters
        ----------
        offset : float from 0 to 1
            The offset of the new color stop.
        color : None or color
            The color of the new stop, or None if the current color at that
            offset is to be used.  The color can be provided in any form that
            an Enable color trait will accept.

        Returns
        -------
        stop : ColorStop
            The new color stop.
        """
        if color is None:
            color = self.color_at(offset)
        stops = self.gradient.stops
        for i, stop in enumerate(stops):
            if stop.offset > offset:
                break
        stop = ColorStop(offset=offset, color=color)
        stops.insert(i, stop)
        return stop

    def color_at(self, offset):
        """ Get the gradient color at an offset.

        Parameters
        ----------
        offset : float from 0 to 1
            The offset of the color.

        Returns
        -------
        color : (r, g, b, a) tuple
            The color tuple at that location.
        """
        stops = self.gradient.stops
        for i, stop in enumerate(stops):
            if stop.offset == offset:
                return stop.color
            if stop.offset > offset:
                break
        else:
            return stops[-1].color

        left_stop = stops[i-1]
        right_stop = stop

        t = (offset - left_stop.offset)/(right_stop.offset - left_stop.offset)
        left_color = np.array(left_stop.color_)
        right_color = np.array(right_stop.color_)
        color = t * right_color + (1 - t) * left_color
        return tuple(color)

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        """ Draw the gradient into the mainlayer.
        """
        with gc:
            if self._brush is not None:
                self._brush.set_brush(gc)
            dx, dy = self.bounds
            x, y = self.position
            gc.rect(x, y, dx, dy)
            gc.fill_path()

    def _gradient_changed(self):
        if self._brush is not None:
            self._brush.gradient = self.gradient

    def __brush_default(self):
        gradient = self.gradient
        brush = LinearGradientBrush(
            start=(0.0, 0.0),
            end=(1.0, 0.0),
            gradient=gradient,
            units="objectBoundingBox",
        )
        return brush

    @observe('+update.updated')
    def _observe_update(self, event):
        self.request_redraw()


class ColorStopOverlay(AbstractOverlay):
    """ An overlay that renders markers at the location of color stops.
    """

    #: The marker to use to mark the stop locations.
    marker = marker_trait("triangle", update=True)

    #: The size of the markers.
    marker_size = Int(5, update=True)

    #: The color of the marker lines.
    marker_line_color = black_color_trait(update=True)

    def overlay(self, other_component, gc, view_bounds=None, mode="normal"):
        """ Draw the overlay.
        """
        marker = self.marker_()
        with gc:
            gc.translate_ctm(
                0,
                self.component.padding_bottom,
            )
            gc.begin_path()
            for stop in self.component.gradient.stops:
                x = self.component.map_screen(stop.offset)
                with gc:
                    gc.translate_ctm(x, 0)
                    gc.set_fill_color(stop.color_)
                    gc.set_stroke_color(self.marker_line_color_)
                    marker.add_to_path(gc, self.marker_size)
                    gc.set_line_width(1.0)
                    if stop == self.component.selected:
                        gc.draw_rect([
                            -self.marker_size/2,
                            0,
                            self.marker_size,
                            self.component.height
                        ])
                    gc.draw_path(marker.draw_mode)

    @observe('component.gradient.updated')
    def _gradient_updated(self, event):
        self.request_redraw()

    @observe('+update')
    def _style_updated(self, event):
        self.request_redraw()


class EditSelectedTool(BaseTool):
    """ Tool that opens a TraitsUI dialog to edit the selected object.
    """

    #: The view to use for the modal dialog.
    edit_view = Instance(View)

    def normal_left_dclick(self, event):
        """ Trigger the edit dialog with a left double-click. """
        if self.component.selected is not None:
            self.component.selected.edit_traits(
                view=self.edit_view,
                kind="livemodal",
            )
            event.handled = True


class DeleteSelectedTool(BaseTool):
    """ Tool that deletes a selected color stop via delete/backspace keys.

    The first and last color stops can't be deleted.
    """

    def normal_key_pressed(self, event):
        """ Process a key press. """
        no_delete = {
            None,
            self.component.gradient.stops[0],
            self.component.gradient.stops[-1],
        }
        delete_keys = {"Backspace", "Delete"}
        selected = self.component.selected
        if selected not in no_delete and event.character in delete_keys:
            self.component.selected = None
            self.component.gradient.stops.remove(selected)
            event.handled = True


class SelectionTool(BaseTool):
    """ Tool that selects a stop if a click hits it.
    """

    def normal_left_down(self, event):
        """ Set the selected stop to the one that is hit. """
        self.component.selected = self.component.hittest(event.x)


class AddStopTool(BaseTool):
    """ Tool adds a new stop on a click if nothing is selected.
    """

    def normal_left_down(self, event):
        """ If nothing is selected, add a stop at the clicked position. """
        if self.component.selected is None:
            offset = self.component.map_data(event.x)
            self.component.selected = self.component.add_stop_at(offset)


class DragSelectedTool(AttributeDragTool):
    """ A tool that handles dragging color stops on a gradient.

    The first and last color stops can't be dragged.
    """

    #: The selected color stop.
    model = Instance(ColorStop)

    #: The attribute to link to the drag.
    x_attr = "offset"

    #: The bounds on the valus of the attribute.
    x_bounds = (0.0, 1.0)

    def is_draggable(self, x, y):
        """ The end stops are not draggable. """
        end_stops = {
            self.component.gradient.stops[0],
            self.component.gradient.stops[-1],
        }
        return (self.model not in end_stops)

    @observe('component.selected')
    def _selection_updated(self, event):
        if self.component is not None:
            self.model = self.component.selected
        else:
            self.model = None


class Demo(DemoFrame):

    def _create_component(self):
        box = GradientBox(
            padding=10,
            gradient=Gradient(
                stops=[
                    ColorStop(offset=0.0, color="red"),
                    ColorStop(offset=0.25, color="yellow"),
                    ColorStop(offset=0.5, color="lime"),
                    ColorStop(offset=0.75, color="cyan"),
                    ColorStop(offset=1.0, color="blue"),
                ],
            ),
        )
        box.overlays.append(
            ColorStopOverlay(component=box)
        )
        # the order of the tools is important for click and keypress handling.
        box.tools.extend([
            SelectionTool(component=box),
            AddStopTool(component=box),
            DragSelectedTool(
                component=box,
                x_mapper=box,  # GradientBox satisfies the mapper interface
            ),
            DeleteSelectedTool(component=box),
            EditSelectedTool(
                component=box,
                edit_view=View(
                    Item('color', style='custom'),
                    title="Edit Gradient Stop",
                    buttons=OKCancelButtons,
                ),
            )
        ])
        return box


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo)
