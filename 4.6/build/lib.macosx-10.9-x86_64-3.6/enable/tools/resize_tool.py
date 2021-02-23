# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from traits.api import Bool, Enum, Int, Set

from ..enable_traits import bounds_trait
from .value_drag_tool import ValueDragTool

hotspot_trait = Enum(
    "top",
    "left",
    "right",
    "bottom",
    "top left",
    "top right",
    "bottom left",
    "bottom right",
)


class ResizeTool(ValueDragTool):
    """ Generic tool for resizing a component
    """

    # Should the resized component be raised to the top of its container's
    # list of components?  This is only recommended for overlaying containers
    # and canvases, but generally those are the only ones in which the
    # ResizeTool will be useful.
    auto_raise = Bool(True)

    #: the hotspots which are active for this tool
    hotspots = Set(hotspot_trait)

    #: the distance in pixels from a hotspot required to register a hit
    threshhold = Int(10)

    #: the minimum bounds that we can resize to
    minimum_bounds = bounds_trait

    #: the hotspot that started the drag
    _selected_hotspot = hotspot_trait

    # 'ValueDragTool' Interface ##############################################

    def get_value(self):
        if self.component is not None:
            c = self.component
            return c.position[:], c.bounds[:]

    def set_delta(self, value, delta_x, delta_y):
        if self.component is not None:
            c = self.component
            position, bounds = value
            x, y = position
            width, height = bounds
            min_width, min_height = self.minimum_bounds
            edges = self._selected_hotspot.split()
            if "left" in edges:
                if delta_x >= width - min_width:
                    delta_x = width - min_width
                c.x = x + delta_x
                c.width = width - delta_x
            if "right" in edges:
                if delta_x <= -width + min_width:
                    delta_x = -width + min_width
                c.width = width + delta_x
            if "bottom" in edges:
                if delta_y >= height - min_height:
                    delta_y = height - min_height
                c.y = y + delta_y
                c.height = height - delta_y
            if "top" in edges:
                if delta_y <= -height + min_height:
                    delta_y = -height + min_height
                c.height = height + delta_y
            c._layout_needed = True
            c.request_redraw()

    # 'DragTool' Interface ###################################################

    def is_draggable(self, x, y):
        return self._find_hotspot(x, y) in self.hotspots

    def drag_start(self, event):
        if self.component is not None:
            self._selected_hotspot = self._find_hotspot(event.x, event.y)
            super(ResizeTool, self).drag_start(event)
            self.component._layout_needed = True
            if self.auto_raise:
                # Push the component to the top of its container's list
                self.component.container.raise_component(self.component)
            event.window.set_mouse_owner(self, event.net_transform())
            event.handled = True
        return True

    # Private Interface ######################################################

    def _find_hotspot(self, x, y):
        hotspot = []
        if self.component is not None:
            c = self.component

            v_threshhold = min(self.threshhold, c.height / 2.0)
            if c.y <= y <= c.y + v_threshhold:
                hotspot.append("bottom")
            elif c.y2 + 1 - v_threshhold <= y <= c.y2 + 1:
                hotspot.append("top")
            elif y < c.y or y > c.y2 + 1:
                return ""

            h_threshhold = min(self.threshhold, c.width / 2.0)
            if c.x <= x <= c.x + h_threshhold:
                hotspot.append("left")
            elif c.x2 + 1 - h_threshhold <= x <= c.x2 + 1:
                hotspot.append("right")
            elif x < c.x or x > c.x2 + 1:
                return ""
        return " ".join(hotspot)

    # Traits Handlers ########################################################

    def _hotspots_default(self):
        return set(
            [
                "top",
                "left",
                "right",
                "bottom",
                "top left",
                "top right",
                "bottom left",
                "bottom right",
            ]
        )

    def _minimum_bounds_default(self):
        return [self.threshhold * 2, self.threshhold * 2]
