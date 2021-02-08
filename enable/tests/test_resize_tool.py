# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import unittest


from enable.component import Component
from enable.tools.resize_tool import ResizeTool


class DragToolTestCase(unittest.TestCase):
    def setUp(self):
        self.component = Component(
            position=[50, 50], bounds=[100, 100], padding=10
        )
        self.tool = ResizeTool(component=self.component)

    def test_find_hotspots(self):
        points_and_results = [
            # corners and edges
            ([50, 50], "bottom left"),
            ([50, 100], "left"),
            ([50, 150], "top left"),
            ([100, 50], "bottom"),
            ([100, 100], ""),
            ([100, 150], "top"),
            ([150, 50], "bottom right"),
            ([150, 100], "right"),
            ([150, 150], "top right"),
            # just inside threshhold
            ([60, 60], "bottom left"),
            ([60, 100], "left"),
            ([60, 140], "top left"),
            ([100, 50], "bottom"),
            ([100, 140], "top"),
            ([140, 60], "bottom right"),
            ([140, 100], "right"),
            ([140, 140], "top right"),
            # just outside box
            ([49, 49], ""),
            ([49, 50], ""),
            ([50, 49], ""),
            ([49, 100], ""),
            ([49, 151], ""),
            ([50, 151], ""),
            ([49, 150], ""),
            ([100, 49], ""),
            ([100, 151], ""),
            ([151, 49], ""),
            ([150, 49], ""),
            ([151, 50], ""),
            ([151, 100], ""),
            ([151, 151], ""),
            ([150, 151], ""),
            ([151, 150], ""),
            # just outside threshhold
            ([61, 61], ""),
            ([60, 61], "left"),
            ([61, 60], "bottom"),
            ([61, 100], ""),
            ([61, 139], ""),
            ([60, 139], "left"),
            ([61, 140], "top"),
            ([100, 61], ""),
            ([100, 139], ""),
            ([139, 61], ""),
            ([140, 61], "right"),
            ([139, 60], "bottom"),
            ([139, 100], ""),
            ([139, 139], ""),
            ([140, 139], "right"),
            ([139, 140], "top"),
        ]
        for (x, y), result in points_and_results:
            value = self.tool._find_hotspot(x, y)
            self.assertEqual(
                value,
                result,
                "Failed at (%f, %f): expected %s, got %s"
                % (x, y, result, value),
            )

    def test_set_delta_left(self):
        self.tool._selected_hotspot = "left"
        value = (self.component.position[:], self.component.bounds[:])
        deltas_and_results = [
            ([10, 10], ([60, 50], [90, 100])),
            ([-10, 10], ([40, 50], [110, 100])),
            ([10, -10], ([60, 50], [90, 100])),
            ([-10, -10], ([40, 50], [110, 100])),
            ([90, 10], ([130, 50], [20, 100])),
            ([80, 10], ([130, 50], [20, 100])),
            ([79, 10], ([129, 50], [21, 100])),
        ]
        for (x, y), (position, bounds) in deltas_and_results:
            self.tool.set_delta(value, x, y)
            self.assertEqual(self.component.position, position)
            self.assertEqual(self.component.bounds, bounds)

    def test_set_delta_right(self):
        self.tool._selected_hotspot = "right"
        value = (self.component.position[:], self.component.bounds[:])
        deltas_and_results = [
            ([10, 10], ([50, 50], [110, 100])),
            ([-10, 10], ([50, 50], [90, 100])),
            ([10, -10], ([50, 50], [110, 100])),
            ([-10, -10], ([50, 50], [90, 100])),
            ([-90, 10], ([50, 50], [20, 100])),
            ([-80, 10], ([50, 50], [20, 100])),
            ([-79, 10], ([50, 50], [21, 100])),
        ]
        for (x, y), (position, bounds) in deltas_and_results:
            self.tool.set_delta(value, x, y)
            self.assertEqual(self.component.position, position)
            self.assertEqual(self.component.bounds, bounds)

    def test_set_delta_bottom(self):
        self.tool._selected_hotspot = "bottom"
        value = (self.component.position[:], self.component.bounds[:])
        deltas_and_results = [
            ([10, 10], ([50, 60], [100, 90])),
            ([-10, 10], ([50, 60], [100, 90])),
            ([10, -10], ([50, 40], [100, 110])),
            ([-10, -10], ([50, 40], [100, 110])),
            ([10, 90], ([50, 130], [100, 20])),
            ([10, 80], ([50, 130], [100, 20])),
            ([10, 79], ([50, 129], [100, 21])),
        ]
        for (x, y), (position, bounds) in deltas_and_results:
            self.tool.set_delta(value, x, y)
            self.assertEqual(self.component.position, position)
            self.assertEqual(self.component.bounds, bounds)

    def test_set_delta_top(self):
        self.tool._selected_hotspot = "top"
        value = (self.component.position[:], self.component.bounds[:])
        deltas_and_results = [
            ([10, 10], ([50, 50], [100, 110])),
            ([-10, 10], ([50, 50], [100, 110])),
            ([10, -10], ([50, 50], [100, 90])),
            ([-10, -10], ([50, 50], [100, 90])),
            ([10, -90], ([50, 50], [100, 20])),
            ([10, -80], ([50, 50], [100, 20])),
            ([10, -79], ([50, 50], [100, 21])),
        ]
        for (x, y), (position, bounds) in deltas_and_results:
            self.tool.set_delta(value, x, y)
            self.assertEqual(self.component.position, position)
            self.assertEqual(self.component.bounds, bounds)
