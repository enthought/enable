# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

import unittest

from pyface.color import Color
from traits.api import HasTraits, TraitError
from traits.testing.optional_dependencies import numpy as np, requires_numpy
from traitsui.api import EditorFactory

from enable.tests._testing import skip_if_null
from enable.trait_defs.rgba_color_trait import RGBAColor


rgba_float_dtype = np.dtype([
    ('red', "float64"),
    ('green', "float64"),
    ('blue', "float64"),
    ('alpha', "float64"),
])
rgba_uint8_dtype = np.dtype([
    ('red', "uint8"),
    ('green', "uint8"),
    ('blue', "uint8"),
    ('alpha', "uint8"),
])
rgb_float_dtype = np.dtype([
    ('red', "float64"),
    ('green', "float64"),
    ('blue', "float64"),
])
rgb_uint8_dtype = np.dtype([
    ('red', "uint8"),
    ('green', "uint8"),
    ('blue', "uint8"),
])


class ColorClass(HasTraits):

    color = RGBAColor()


class TestRGBAColor(unittest.TestCase):

    def test_init(self):
        trait = RGBAColor()
        self.assertEqual(trait.default_value, (1.0, 1.0, 1.0, 1.0))

    @requires_numpy
    def test_default_value(self):
        values = [
            "rebeccapurple",
            "rebecca purple",
            "#666633339999ffff",
            "#663399ff",
            "#639f",
            Color(rgba=(0.4, 0.2, 0.6, 1.0)),
            Color(rgba=(0.4, 0.2, 0.6, 1.0)).to_toolkit(),
            (0.4, 0.2, 0.6, 1.0),
            [0.4, 0.2, 0.6, 1.0],
            np.array([0.4, 0.2, 0.6, 1.0]),
            np.array([(0.4, 0.2, 0.6, 1.0)], dtype=rgba_float_dtype)[0],
            (0x66, 0x33, 0x99, 0xff),
            [0x66, 0x33, 0x99, 0xff],
            np.array([0x66, 0x33, 0x99, 0xff], dtype='uint8'),
            np.array([(0x66, 0x33, 0x99, 0xff)], dtype=rgba_uint8_dtype)[0],
            "#666633339999",
            "#663399",
            "#639",
            0x663399,
            (0.4, 0.2, 0.6),
            [0.4, 0.2, 0.6],
            np.array([0.4, 0.2, 0.6]),
            np.array([(0.4, 0.2, 0.6)], dtype=rgb_float_dtype)[0],
            (0x66, 0x33, 0x99),
            [0x66, 0x33, 0x99],
            np.array([0x66, 0x33, 0x99], dtype='uint8'),
            np.array([(0x66, 0x33, 0x99)], dtype=rgb_uint8_dtype)[0],
        ]
        for value in values:
            with self.subTest(value=value):
                trait = RGBAColor(value)
                self.assertEqual(trait.default_value, (0.4, 0.2, 0.6, 1.0))

    def test_init_invalid(self):
        values = [
            (0.4, 0.2),
            (0.4, 0.2, 0.3, 1.0, 1.0),
            "notacolor",
            "#66666",
            (0.0, 1.00001, 0.9, 1.0),
            (0.0, -0.00001, 0.9, 1.0),
            (0, -1, 250, 255),
            None,
        ]
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(Exception):
                    RGBAColor(value)

    def test_validate(self):
        values = [
            "rebeccapurple",
            "rebecca purple",
            "#666633339999ffff",
            "#663399ff",
            "#639f",
            Color(rgba=(0.4, 0.2, 0.6, 1.0)),
            Color(rgba=(0.4, 0.2, 0.6, 1.0)).to_toolkit(),
            (0.4, 0.2, 0.6, 1.0),
            [0.4, 0.2, 0.6, 1.0],
            np.array([0.4, 0.2, 0.6, 1.0]),
            np.array([(0.4, 0.2, 0.6, 1.0)], dtype=rgba_float_dtype)[0],
            (0x66, 0x33, 0x99, 0xff),
            [0x66, 0x33, 0x99, 0xff],
            np.array([0x66, 0x33, 0x99, 0xff], dtype='uint8'),
            np.array([(0x66, 0x33, 0x99, 0xff)], dtype=rgba_uint8_dtype)[0],
            "#666633339999",
            "#663399",
            "#639",
            0x663399,
            (0.4, 0.2, 0.6),
            [0.4, 0.2, 0.6],
            np.array([0.4, 0.2, 0.6]),
            np.array([(0.4, 0.2, 0.6)], dtype=rgb_float_dtype)[0],
            (0x66, 0x33, 0x99),
            [0x66, 0x33, 0x99],
            np.array([0x66, 0x33, 0x99], dtype='uint8'),
            np.array([(0x66, 0x33, 0x99)], dtype=rgb_uint8_dtype)[0],
        ]
        trait = RGBAColor()
        for value in values:
            with self.subTest(value=value):
                validated = trait.validate(None, None, value)
                self.assertEqual(validated, (0.4, 0.2, 0.6, 1.0))

    def test_validate_invalid(self):
        values = [
            (0.4, 0.2),
            (0.4, 0.2, 0.3, 1.0, 1.0),
            "notacolor",
            "#66666",
            (0.0, 1.00001, 0.9, 1.0),
            (0.0, -0.00001, 0.9, 1.0),
            (0, -1, 250, 255),
            None,
        ]
        trait = RGBAColor()
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(TraitError):
                    trait.validate(None, None, value)

    def test_info(self):
        trait = RGBAColor()
        self.assertIsInstance(trait.info(), str)

    def test_default_trait(self):
        color_class = ColorClass()
        self.assertEqual(color_class.color, (1.0, 1.0, 1.0, 1.0))

    def test_trait_set(self):
        values = [
            "rebeccapurple",
            "rebecca purple",
            "#666633339999ffff",
            "#663399ff",
            "#639f",
            Color(rgba=(0.4, 0.2, 0.6, 1.0)),
            Color(rgba=(0.4, 0.2, 0.6, 1.0)).to_toolkit(),
            (0.4, 0.2, 0.6, 1.0),
            [0.4, 0.2, 0.6, 1.0],
            np.array([0.4, 0.2, 0.6, 1.0]),
            np.array([(0.4, 0.2, 0.6, 1.0)], dtype=rgba_float_dtype)[0],
            (0x66, 0x33, 0x99, 0xff),
            [0x66, 0x33, 0x99, 0xff],
            np.array([0x66, 0x33, 0x99, 0xff], dtype='uint8'),
            np.array([(0x66, 0x33, 0x99, 0xff)], dtype=rgba_uint8_dtype)[0],
            "#666633339999",
            "#663399",
            "#639",
            0x663399,
            (0.4, 0.2, 0.6),
            [0.4, 0.2, 0.6],
            np.array([0.4, 0.2, 0.6]),
            np.array([(0.4, 0.2, 0.6)], dtype=rgb_float_dtype)[0],
            (0x66, 0x33, 0x99),
            [0x66, 0x33, 0x99],
            np.array([0x66, 0x33, 0x99], dtype='uint8'),
            np.array([(0x66, 0x33, 0x99)], dtype=rgb_uint8_dtype)[0],
        ]
        for value in values:
            with self.subTest(value=value):
                color_class = ColorClass(color=value)
                self.assertEqual(color_class.color, (0.4, 0.2, 0.6, 1.0))

    def test_trait_set_invalid(self):
        values = [
            (0.4, 0.2),
            (0.4, 0.2, 0.3, 1.0, 1.0),
            "notacolor",
            "#66666",
            (0.0, 1.00001, 0.9, 1.0),
            (0.0, -0.00001, 0.9, 1.0),
            (0, -1, 250, 255),
            None,
        ]
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(TraitError):
                    ColorClass(color=value)

    @skip_if_null
    def test_get_editor(self):
        trait = RGBAColor()
        editor = trait.get_editor()

        self.assertIsInstance(editor, EditorFactory)

    def test_sys_window_color(self):
        trait = RGBAColor()
        # smoke-test: value depends on system and user preferences
        trait.validate(None, None, "syswindow")
        # older code used with an underscore is also OK
        trait.validate(None, None, "sys_window")
