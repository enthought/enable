# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Tests for the brushes """

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from kiva.celiagg import GraphicsContext
from kiva.testing import KivaTestAssistant
from traits.testing.api import UnittestTools

from enable.brush import (
    ColorBrush, ColorStop, Gradient, LinearGradientBrush, RadialGradientBrush
)


class BrushTestMixin(KivaTestAssistant, UnittestTools):

    def create_brush(self):
        raise NotImplementedError()

    def do_draw_with_brush(self, gc):
        brush = self.create_brush()
        with gc:
            brush.set_brush(gc)
            gc.rect(50, 50, 100, 100)
            gc.fill_path()

    def do_update_brush_trait(self, trait, value):
        brush = self.create_brush()
        with self.assertTraitChanges(brush, "updated", count=1):
            setattr(brush, trait, value)

    def test_draw_celiagg(self):
        gc = GraphicsContext((200, 200))
        self.do_draw_with_brush(gc)


class GradientBrushTestMixin(BrushTestMixin):

    def set_brush_stops(self, brush):
        brush.gradient.stops = [
            ColorStop(offset=0.0, color='red'),
            ColorStop(offset=0.5, color='yellow'),
            ColorStop(offset=1.0, color='lime'),
        ]

    def test_spread_method_updated(self):
        self.do_update_brush_trait("spread_method", "reflect")

    def test_units_updated(self):
        self.do_update_brush_trait("spread_method", "reflect")

    def test_gradient_updated(self):
        self.do_update_brush_trait("gradient", Gradient())


class TestColorBrush(TestCase, BrushTestMixin):

    def create_brush(self):
        return ColorBrush(color="red")

    def test_draw_mock(self):
        gc = self.create_mock_gc(200, 200, ["set_fill_color"])
        self.do_draw_with_brush(gc)

        gc.set_fill_color.assert_called_once_with((1.0, 0.0, 0.0, 1.0))

    def test_color_updated(self):
        self.do_update_brush_trait("color", "blue")


class TestColorStop(TestCase, UnittestTools):

    def test_to_array(self):
        color_stop = ColorStop(offset=0.5, color="red")

        a = color_stop.to_array()

        assert_array_equal(a, np.array([0.5, 1.0, 0.0, 0.0, 1.0]))

    def test_offset_updated(self):
        color_stop = ColorStop(offset=0.5, color="red")

        with self.assertTraitChanges(color_stop, "updated", count=1):
            color_stop.offset = 0.25

    def test_color_updated(self):
        color_stop = ColorStop(offset=0.5, color="red")

        with self.assertTraitChanges(color_stop, "updated", count=1):
            color_stop.color = "blue"


class TestGradient(TestCase, UnittestTools):

    def create_gradient(self):
        return Gradient()

    def set_stops(self, gradient):
        gradient.stops = [
            ColorStop(offset=0.0, color='red'),
            ColorStop(offset=0.5, color='yellow'),
            ColorStop(offset=1.0, color='lime'),
        ]

    def test_default_draw_celiagg(self):
        gc = GraphicsContext((200, 200))
        gradient = self.create_gradient()
        with gc:
            gc.linear_gradient(
                125,
                75,
                75,
                125,
                gradient.to_array(),
                "pad",
            )

    def test_to_array(self):
        gradient = self.create_gradient()
        self.set_stops(gradient)

        a = gradient.to_array()

        assert_array_equal(
            a,
            np.array([
                np.array([0.0, 1.0, 0.0, 0.0, 1.0]),
                np.array([0.5, 1.0, 1.0, 0.0, 1.0]),
                np.array([1.0, 0.0, 1.0, 0.0, 1.0]),
            ])
        )

    def test_stops_updated(self):
        gradient = self.create_gradient()
        self.set_stops(gradient)
        with self.assertTraitChanges(gradient, "updated", count=1):
            gradient.stops = [
                ColorStop(offset=0.0, color='blue'),
                ColorStop(offset=1.0, color='lime'),
            ]

    def test_stops_items_updated(self):
        gradient = self.create_gradient()
        self.set_stops(gradient)
        with self.assertTraitChanges(gradient, "updated", count=1):
            gradient.stops[0] = ColorStop(offset=0.0, color='blue')

    def test_stops_items_updated_updated(self):
        gradient = self.create_gradient()
        self.set_stops(gradient)
        with self.assertTraitChanges(gradient, "updated", count=1):
            gradient.stops[0].updated = True


class TestLinearGradientBrush(TestCase, GradientBrushTestMixin):

    def create_brush(self):
        brush = LinearGradientBrush(start=(125, 75), end=(75, 125))
        self.set_brush_stops(brush)
        return brush

    def test_draw_mock(self):
        gc = self.create_mock_gc(200, 200, ["linear_gradient"])
        self.do_draw_with_brush(gc)

        gc.linear_gradient.assert_called_once()

        args, kw = gc.linear_gradient.call_args
        self.assertEqual(kw, {})

        x1, y1, x2, y2, stops, spread, units = args

        self.assertEqual((x1, y1, x2, y2), (125, 75, 75, 125))
        assert_array_equal(
            stops,
            np.array([
                np.array([0.0, 1.0, 0.0, 0.0, 1.0]),
                np.array([0.5, 1.0, 1.0, 0.0, 1.0]),
                np.array([1.0, 0.0, 1.0, 0.0, 1.0]),
            ])
        )
        self.assertEqual(spread, "pad")
        self.assertEqual(units, "userSpaceOnUse")

    def test_start_updated(self):
        self.do_update_brush_trait("start", (100, 100))

    def test_end_updated(self):
        self.do_update_brush_trait("end", (100, 100))


class TestRadialGradientBrush(TestCase, GradientBrushTestMixin):

    def create_brush(self):
        brush = RadialGradientBrush(
            center=(125, 75),
            radius=100,
            focus=(75, 100),
        )
        self.set_brush_stops(brush)
        return brush

    def test_draw_mock(self):
        gc = self.create_mock_gc(200, 200, ["radial_gradient"])
        self.do_draw_with_brush(gc)

        gc.radial_gradient.assert_called_once()

        args, kw = gc.radial_gradient.call_args
        self.assertEqual(kw, {})

        x1, y1, r, x2, y2, stops, spread, units = args

        self.assertEqual((x1, y1, x2, y2), (125, 75, 75, 100))
        self.assertEqual(r, 100)
        assert_array_equal(
            stops,
            np.array([
                np.array([0.0, 1.0, 0.0, 0.0, 1.0]),
                np.array([0.5, 1.0, 1.0, 0.0, 1.0]),
                np.array([1.0, 0.0, 1.0, 0.0, 1.0]),
            ])
        )
        self.assertEqual(spread, "pad")
        self.assertEqual(units, "userSpaceOnUse")

    def test_center_updated(self):
        self.do_update_brush_trait("center", (100, 100))

    def test_radius_updated(self):
        self.do_update_brush_trait("radius", 150)

    def test_focus_updated(self):
        self.do_update_brush_trait("focus", (100, 100))
