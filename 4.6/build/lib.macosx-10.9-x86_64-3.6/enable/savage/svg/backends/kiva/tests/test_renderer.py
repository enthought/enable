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

from enable.savage.svg.backends.kiva.renderer import (
    LinearGradientBrush,
    RadialGradientBrush,
    Renderer,
)


class TestRenderer(unittest.TestCase):
    def test_linear_gradient_brush(self):
        lgb = LinearGradientBrush(1, 1, 2, 2, 3)
        lgb.transforms.append("a")
        self.assertEqual(LinearGradientBrush(1, 1, 2, 2, 3).transforms, [])

    def test_radial_gradient_brush(self):
        rgb = RadialGradientBrush(1, 1, 2, 2, 3)
        rgb.transforms.append("a")
        self.assertEqual(RadialGradientBrush(1, 1, 2, 2, 3).transforms, [])

    def test_create_linear_gradient_brush(self):
        renderer = Renderer()
        lgb = renderer.createLinearGradientBrush(1, 1, 2, 2, 3)
        lgb.transforms.append("a")
        self.assertEqual(
            renderer.createLinearGradientBrush(1, 1, 2, 2, 3).transforms, []
        )

    def test_create_radial_gradient_brush(self):
        renderer = Renderer()
        rgb = renderer.createRadialGradientBrush(1, 1, 2, 2, 3)
        rgb.transforms.append("a")
        self.assertEqual(
            renderer.createRadialGradientBrush(1, 1, 2, 2, 3).transforms, []
        )
