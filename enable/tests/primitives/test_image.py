# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Tests for the Image component """

import os
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from pkg_resources import resource_filename

from kiva.image import GraphicsContext
from traits.api import TraitError
from traits.testing.api import UnittestTools

from enable.primitives.image import Image


data_dir = resource_filename("enable.tests.primitives", "data")


class ImageTest(unittest.TestCase, UnittestTools):
    def setUp(self):
        self.data = np.empty(shape=(128, 256, 4), dtype="uint8")
        self.data[:, :, 0] = np.arange(256)
        self.data[:, :, 1] = np.arange(128)[:, np.newaxis]
        self.data[:, :, 2] = np.arange(256)[::-1]
        self.data[:, :, 3] = np.arange(128)[::-1, np.newaxis]

        self.image_24 = Image(self.data[..., :3])
        self.image_32 = Image(self.data)

    def test_fromfile_png_rgb(self):
        # basic smoke test - assume that kiva.image does the right thing
        path = os.path.join(data_dir, "PngSuite", "basn2c08.png")
        image = Image.from_file(path)

        self.assertEqual(image.data.shape, (32, 32, 3))
        self.assertEqual(image.format, "rgb24")

    def test_fromfile_png_rgba(self):
        # basic smoke test - assume that kiva.image does the right thing
        path = os.path.join(data_dir, "PngSuite", "basi6a08.png")
        image = Image.from_file(path)

        self.assertEqual(image.data.shape, (32, 32, 4))
        self.assertEqual(image.format, "rgba32")

    def test_init_bad_shape(self):
        data = np.zeros(shape=(256, 256), dtype="uint8")
        with self.assertRaises(TraitError):
            Image(data=data)

    def test_init_bad_dtype(self):
        data = np.array(["red"] * 65536).reshape(128, 128, 4)
        with self.assertRaises(TraitError):
            Image(data=data)

    def test_set_bad_shape(self):
        data = np.zeros(shape=(256, 256), dtype="uint8")
        with self.assertRaises(TraitError):
            self.image_32.data = data

    def test_set_bad_dtype(self):
        data = np.array(["red"] * 65536).reshape(128, 128, 4)
        with self.assertRaises(TraitError):
            self.image_32.data = data

    def test_format(self):
        self.assertEqual(self.image_24.format, "rgb24")
        self.assertEqual(self.image_32.format, "rgba32")

    def test_format_change(self):
        image = self.image_24
        with self.assertTraitChanges(image, "format"):
            image.data = self.data

        self.assertEqual(self.image_24.format, "rgba32")

    def test_bounds_default(self):
        self.assertEqual(self.image_24.bounds, [256, 128])
        self.assertEqual(self.image_32.bounds, [256, 128])

    def test_bounds_overrride(self):
        image = Image(self.data, bounds=[200, 100])
        self.assertEqual(image.bounds, [200, 100])

    def test_size_hint(self):
        self.assertEqual(self.image_24.layout_size_hint, (256, 128))
        self.assertEqual(self.image_32.layout_size_hint, (256, 128))

    def test_size_hint_change(self):
        data = np.zeros(shape=(256, 128, 3), dtype="uint8")
        image = self.image_24
        with self.assertTraitChanges(image, "layout_size_hint"):
            image.data = data

        self.assertEqual(self.image_24.layout_size_hint, (128, 256))

    def test_image_gc_24(self):
        # this is non-contiguous, because data comes from slice
        image = self.image_24._image
        assert_array_equal(image, self.data[..., :3])

    def test_image_gc_32(self):
        # this is contiguous
        image = self.image_32._image
        assert_array_equal(image, self.data)

    def test_draw_24(self):
        gc = GraphicsContext((256, 128), pix_format="rgb24")
        self.image_24.draw(gc)
        # if test is failing, uncomment this line to see what is drawn
        # gc.save('test_image_draw_24.png')

        # smoke test: image isn't all white
        assert_array_equal(gc.bmp_array[..., :3], self.data[..., :3])

        gc2 = GraphicsContext((256, 128), pix_format="rgba32")
        self.image_24.draw(gc2)
        assert_array_equal(gc2.bmp_array[..., :3], self.data[..., :3])

    def test_draw_32(self):
        gc = GraphicsContext((256, 128), pix_format="rgba32")
        self.image_32.draw(gc)
        # if test is failing, uncommetn this line to see what is drawn
        # gc.save('test_image_draw_32.png')

        # smoke test: image isn't all white
        # XXX actually compute what it should look like with alpha transfer
        white_image = np.ones(shape=(256, 128, 4), dtype="uint8") * 255
        self.assertFalse(np.array_equal(white_image, gc.bmp_array))

    def test_draw_stretched(self):
        gc = GraphicsContext((256, 256), pix_format="rgba32")
        self.image_32.bounds = [128, 258]
        self.image_32.position = [128, 0]
        self.image_32.draw(gc)
        # if test is failing, uncommetn this line to see what is drawn
        # gc.save('test_image_draw_stretched.png')

        # smoke test: image isn't all white
        # XXX actually compute what it should look like with alpha transfer
        white_image = np.ones(shape=(256, 256, 4), dtype="uint8") * 255
        self.assertFalse(np.array_equal(white_image, gc.bmp_array))

        # left half of the image *should* be white
        assert_array_equal(gc.bmp_array[:, :128, :], white_image[:, :128, :])
