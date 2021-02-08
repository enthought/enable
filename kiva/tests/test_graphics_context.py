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

from hypothesis import given
from hypothesis.strategies import sampled_from
from numpy import (
    alltrue, array, asarray, concatenate, dtype, newaxis, ones, ravel, zeros
)
from PIL import Image as PILImage


from kiva.image import GraphicsContext

# alpha blending is approximate in agg, so we allow some "slop" between
# desired and actual results, allow channel differences of to 2.
slop_allowed = 2

UInt8 = dtype("uint8")
Int32 = dtype("int32")


class TestAlphaBlackImage(unittest.TestCase):
    def setUp(self):
        self.size = (1, 1)
        self.color = 0.0

    @given(sampled_from([1.0, 0.0, 0.5]))
    def test_simple(self, color):
        gc = GraphicsContext(self.size, pix_format="bgra32")
        desired = self.solid_bgra32(self.size, color)
        img = GraphicsContext(desired, pix_format="bgra32")
        gc.draw_image(img)
        actual = gc.bmp_array
        # for alpha == 1, image should be exactly equal.
        self.assert_images_equal(desired, actual)

    @given(sampled_from([1.0, 0.0, 0.5]))
    def test_image_alpha(self, color):
        alpha = 0.5
        gc = GraphicsContext(self.size, pix_format="bgra32")
        orig = self.solid_bgra32(self.size, color, alpha)
        img = GraphicsContext(orig, pix_format="bgra32")
        gc.draw_image(img)
        actual = gc.bmp_array
        gc_background = self.solid_bgra32(self.size, 1.0)
        orig = self.solid_bgra32(self.size, color, alpha)
        desired = self.alpha_blend(gc_background, orig)

        # also, the alpha channel of the image is not copied into the
        # desination graphics context, so we have to ignore alphas
        self.assert_images_close(
            desired[:, :, :-1], actual[:, :, :-1], diff_allowed=slop_allowed
        )

    @given(sampled_from([1.0, 0.0, 0.5]))
    def test_ambient_alpha(self, color):
        orig = self.solid_bgra32(self.size, color)
        img = GraphicsContext(orig, pix_format="bgra32")
        gc = GraphicsContext(self.size, pix_format="bgra32")
        amb_alpha = 0.5
        gc.set_alpha(amb_alpha)
        gc.draw_image(img)
        actual = gc.bmp_array

        gc_background = self.solid_bgra32(self.size, 1.0)
        orig = self.solid_bgra32(self.size, color)
        desired = self.alpha_blend(
            gc_background, orig, ambient_alpha=amb_alpha
        )
        # alpha blending is approximate, allow channel differences of to 2.
        self.assert_images_close(desired, actual, diff_allowed=slop_allowed)

    @given(sampled_from([1.0, 0.0, 0.5]))
    def test_ambient_plus_image_alpha(self, color):
        amb_alpha = 0.5
        img_alpha = 0.5
        gc = GraphicsContext(self.size, pix_format="bgra32")
        orig = self.solid_bgra32(self.size, color, img_alpha)
        img = GraphicsContext(orig, pix_format="bgra32")
        gc.set_alpha(amb_alpha)
        gc.draw_image(img)
        actual = gc.bmp_array

        gc_background = self.solid_bgra32(self.size, 1.0)
        orig = self.solid_bgra32(self.size, color, img_alpha)
        desired = self.alpha_blend(
            gc_background, orig, ambient_alpha=amb_alpha
        )
        # alpha blending is approximate, allow channel differences of to 2.
        self.assert_images_close(desired, actual, diff_allowed=slop_allowed)

    def test_rect_scale(self):
        color = 0.0
        orig_sz = (10, 10)
        img_ary = self.solid_bgra32(orig_sz, color)
        orig = GraphicsContext(img_ary, pix_format="bgra32")
        sx, sy = 5, 20
        scaled_rect = (0, 0, sx, sy)
        gc = GraphicsContext((20, 20), pix_format="bgra32")
        gc.draw_image(orig, scaled_rect)
        actual = gc.bmp_array
        desired_sz = (sx, sy)
        img_ary = self.solid_bgra32(desired_sz, color)
        img = GraphicsContext(img_ary, pix_format="bgra32")
        gc = GraphicsContext((20, 20), pix_format="bgra32")
        gc.draw_image(img)
        desired = gc.bmp_array
        self.assert_images_equal(desired, actual)

    def test_rect_scale_translate(self):
        color = 0.0
        orig_sz = (10, 10)
        img_ary = self.solid_bgra32(orig_sz, color)
        orig = GraphicsContext(img_ary, pix_format="bgra32")
        tx, ty = 5, 10
        sx, sy = 5, 20
        translate_scale_rect = (tx, ty, sx, sy)
        gc = GraphicsContext((40, 40), pix_format="bgra32")
        gc.draw_image(orig, translate_scale_rect)
        actual = gc.bmp_array
        desired_sz = (sx, sy)
        img_ary = self.solid_bgra32(desired_sz, color)
        img = GraphicsContext(img_ary, pix_format="bgra32")
        gc = GraphicsContext((40, 40), pix_format="bgra32")
        gc.translate_ctm(tx, ty)
        gc.draw_image(img)
        desired = gc.bmp_array
        self.assert_images_equal(desired, actual)

    def sun(self, interpolation_scheme="simple"):
        pil_img = PILImage.open("doubleprom_soho_full.jpg")
        img = asarray(pil_img)
        alpha = ones(pil_img.size, UInt8) * 255
        img = concatenate((img[:, :, ::-1], alpha[:, :, newaxis]), -1).copy()
        return GraphicsContext(img, "bgra32", interpolation_scheme)

    def alpha_blend(self, src1, src2, alpha=1.0, ambient_alpha=1.0):
        alpha_ary = src2[:, :, 3] / 255.0 * alpha * ambient_alpha
        res = src1[:, :, :] * (1 - alpha_ary) + src2[:, :, :] * alpha_ary
        # alpha blending preserves the alpha mask channel of the
        # destination (src1)
        res[:, :, -1] = src1[:, :, -1]
        return res.astype(Int32)

    def solid_bgra32(self, size, value=0.0, alpha=1.0):
        img_array = zeros((size[1], size[0], 4), UInt8)
        img_array[:, :, :-1] = array(value * 255, UInt8)
        img_array[:, :, -1] = array(alpha * 255, UInt8)
        return img_array

    def assert_images_equal(self, desired, actual):
        """ Only use for small arrays. """
        try:
            assert alltrue(ravel(actual) == ravel(desired))
        except AssertionError:
            size = sum(array(desired.shape))
            if size < 10:
                diff = abs(
                    ravel(actual.astype(Int32)) - ravel(desired.astype(Int32))
                )
                msg = "\n"
                msg += "desired: %s\n" % ravel(desired)
                msg += "actual: %s\n" % ravel(actual)
                msg += "abs diff: %s\n" % diff
            else:
                msg = "size: %d.  To large to display" % size
            raise AssertionError(msg)

    def assert_images_close(self, desired, actual, diff_allowed=2):
        """ Only use for small arrays. """
        try:
            # cast up so math doesn't underflow
            diff = abs(
                ravel(actual.astype(Int32)) - ravel(desired.astype(Int32))
            )
            assert alltrue(diff <= diff_allowed)
        except AssertionError:
            size = sum(array(desired.shape))
            if size < 10:
                msg = "\n"
                msg += "desired: %s\n" % ravel(desired)
                msg += "actual: %s\n" % ravel(actual)
                msg += "abs diff: %s\n" % diff
            else:
                msg = "size: %d.  To large to display" % size
            raise AssertionError(msg)
