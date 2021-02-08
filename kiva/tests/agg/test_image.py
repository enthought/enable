# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import os
import tempfile
from timeit import Timer
import unittest

from numpy import (
    alltrue, array, concatenate, dtype, frombuffer, newaxis, ones, pi, ravel,
    zeros,
)
from PIL import Image

from kiva import agg
from kiva.api import Font


# alpha blending is approximate in agg, so we allow some "slop" between
# desired and actual results, allow channel differences of to 2.
slop_allowed = 2

UInt8 = dtype("uint8")
Int32 = dtype("int32")


def save(img):
    """ This only saves the rgb channels of the image
    """
    imgformat = img.format()
    if imgformat == "bgra32":
        bgr = img.bmp_array[:, :, :3]
        rgb = bgr[:, :, ::-1].copy()
        pil_img = Image.fromarray(rgb, "RGB")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'img.bmp')
            pil_img.save(path)
    else:
        raise NotImplementedError(
            "currently only supports writing out bgra32 images"
        )


def sun(interpolation_scheme="simple"):
    path = os.path.join(os.path.dirname(__file__), "doubleprom_soho_full.jpg")
    pil_img = Image.open(path)
    img = frombuffer(pil_img.tobytes(), UInt8)
    img.resize((pil_img.size[1], pil_img.size[0], 3))

    alpha = ones(pil_img.size, UInt8) * 255
    img = concatenate((img[:, :, ::-1], alpha[:, :, newaxis]), -1).copy()
    return agg.GraphicsContextArray(img, "bgra32", interpolation_scheme)


def solid_bgra32(size, value=0.0, alpha=1.0):
    img_array = zeros((size[1], size[0], 4), UInt8)
    img_array[:, :, :-1] = array(value * 255, UInt8)
    img_array[:, :, -1] = array(alpha * 255, UInt8)
    return img_array


def alpha_blend(src1, src2, alpha=1.0, ambient_alpha=1.0):
    alpha_ary = src2[:, :, 3] / 255.0 * alpha * ambient_alpha
    res = src1[:, :, :] * (1 - alpha_ary) + src2[:, :, :] * alpha_ary
    # alpha blending preserves the alpha mask channel of the destination (src1)
    res[:, :, -1] = src1[:, :, -1]
    return res.astype(Int32)


def assert_equal(desired, actual):
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


def assert_close(desired, actual, diff_allowed=2):
    """ Only use for small arrays. """
    try:
        # cast up so math doesn't underflow
        diff = abs(ravel(actual.astype(Int32)) - ravel(desired.astype(Int32)))
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


# ----------------------------------------------------------------------------
# Tests speed of various interpolation schemes
#
# ----------------------------------------------------------------------------


class test_text_image(unittest.TestCase):
    def test_antialias(self):
        gc = agg.GraphicsContextArray((200, 50), pix_format="bgra32")
        gc.set_antialias(1)
        f = Font("modern")
        gc.set_font(f)
        gc.show_text("hello")
        save(gc)

    def test_no_antialias(self):
        gc = agg.GraphicsContextArray((200, 50), pix_format="bgra32")
        f = Font("modern")
        gc.set_font(f)
        gc.set_antialias(0)
        gc.show_text("hello")
        save(gc)

    def test_rotate(self):
        text = "hello"
        gc = agg.GraphicsContextArray((150, 150), pix_format="bgra32")
        f = Font("modern")
        gc.set_font(f)
        tx, ty, sx, sy = gc.get_text_extent(text)
        gc.translate_ctm(25, 25)
        gc.rotate_ctm(pi / 2.0)
        gc.translate_ctm(0, -sy)
        # gc.show_text(text)
        gc.set_stroke_color([1, 0, 0])
        gc.set_fill_color([0.5, 0.5, 0.5])
        gc.rect(tx, ty, sx, sy)
        gc.stroke_path()
        gc.show_text(text)
        save(gc)


class test_sun(unittest.TestCase):
    def generic_sun(self, scheme):
        img = sun(scheme)
        sz = array((img.width(), img.height()))
        scaled_sz = sz * 0.3
        scaled_rect = (0, 0, scaled_sz[0], scaled_sz[1])
        gc = agg.GraphicsContextArray(tuple(scaled_sz), pix_format="bgra32")
        gc.draw_image(img, scaled_rect)
        return gc

    def test_simple(self):
        gc = self.generic_sun("nearest")
        save(gc)

    def test_bilinear(self):
        gc = self.generic_sun("bilinear")
        save(gc)

    def test_bicubic(self):
        gc = self.generic_sun("bicubic")
        save(gc)

    def test_spline16(self):
        gc = self.generic_sun("spline16")
        save(gc)

    def test_spline36(self):
        gc = self.generic_sun("spline36")
        save(gc)

    def test_sinc64(self):
        gc = self.generic_sun("sinc64")
        save(gc)

    def test_sinc144(self):
        gc = self.generic_sun("sinc144")
        save(gc)

    def test_sinc256(self):
        gc = self.generic_sun("sinc256")
        save(gc)

    def test_blackman100(self):
        gc = self.generic_sun("blackman100")
        save(gc)

    def test_blackman256(self):
        gc = self.generic_sun("blackman256")
        save(gc)


def bench(stmt="pass", setup="pass", repeat=5, adjust_runs=True):
    """ BenchMark the function.
    """
    timer = Timer(stmt, setup)
    if adjust_runs:
        for i in range(100):
            number = 10 ** i
            time = timer.timeit(number)
            if time > 0.02:
                break
    else:
        number = 1
    times = [timer.timeit(number) for i in range(repeat)]
    message = "{} calls, best of {} repeats: {:f} sec per call"
    return message.format(number, repeat, min(times) / number)


# ----------------------------------------------------------------------------
# Tests speed of various interpolation schemes
#
#
# ----------------------------------------------------------------------------
class test_interpolation_image(unittest.TestCase):
    size = (1000, 1000)
    color = 0.0

    def generic_timing(self, scheme, size):
        gc = agg.GraphicsContextArray(size, pix_format="bgra32")
        desired = solid_bgra32(size, self.color)
        img = agg.GraphicsContextArray(
            desired, pix_format="bgra32", interpolation=scheme
        )
        print(
            "{!r} interpolation, ".format(scheme),
            bench(lambda: gc.draw_image(img)),
        )

    def test_simple_timing(self):
        scheme = "nearest"
        self.generic_timing(scheme, self.size)

    def test_bilinear_timing(self):
        scheme = "bilinear"
        self.generic_timing(scheme, self.size)

    def test_bicubic_timing(self):
        scheme = "bicubic"
        self.generic_timing(scheme, self.size)

    def test_sinc144_timing(self):
        scheme = "sinc144"
        self.generic_timing(scheme, self.size)
