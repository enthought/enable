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
