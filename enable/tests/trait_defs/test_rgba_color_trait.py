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

import numpy as np

from pyface.color import Color
from traits.api import DefaultValue, HasTraits, TraitError
from traits.testing.optional_dependencies import numpy as np, requires_numpy
from traitsui.api import EditorFactory

from enable.trait_defs.rgba_color_trait import RGBAColor


class ColorClass(HasTraits):

    color = RGBAColor()


class TestRGBAColor(unittest.TestCase):

    def test_init(self):
        trait = RGBAColor()
        self.assertEqual(trait.default_value, (1.0, 1.0, 1.0, 1.0))

    def test_init_name(self):
        trait = RGBAColor("rebeccapurple")
        self.assertEqual(
            trait.default_value,
            (0.4, 0.2, 0.6, 1.0),
        )

    def test_init_hex(self):
        trait = RGBAColor("#663399ff")
        self.assertEqual(
            trait.default_value,
            (0.4, 0.2, 0.6, 1.0)
        )

    def test_init_color(self):
        trait = RGBAColor(Color(rgba=(0.4, 0.2, 0.6, 1.0)))
        self.assertEqual(
            trait.default_value,
            (0.4, 0.2, 0.6, 1.0)
        )

    def test_init_tuple(self):
        trait = RGBAColor((0.4, 0.2, 0.6, 1.0))
        self.assertEqual(
            trait.default_value,
            (0.4, 0.2, 0.6, 1.0)
        )

    def test_init_list(self):
        trait = RGBAColor([0.4, 0.2, 0.6, 1.0])
        self.assertEqual(
            trait.default_value,
            (0.4, 0.2, 0.6, 1.0)
        )

    def test_init_array(self):
        trait = RGBAColor(np.array([0.4, 0.2, 0.6, 1.0]))
        self.assertEqual(
            trait.default_value,
            (0.4, 0.2, 0.6, 1.0)
        )

    def test_init_array_structured_dtype(self):
        """ Test if "typical" RGBA structured array value works. """
        arr = np.array(
            [(0.4, 0.2, 0.6, 1.0)],
            dtype=np.dtype([
                ('red', float),
                ('green', float),
                ('blue', float),
                ('alpha', float),
            ]),
        )
        trait = RGBAColor(arr[0])
        self.assertEqual(
            trait.default_value,
            (0.4, 0.2, 0.6, 1.0)
        )

    def test_init_invalid(self):
        with self.assertRaises(TraitError):
            RGBAColor((0.4, 0.2))

    def test_validate_color(self):
        color = (0.4, 0.2, 0.6, 1.0)
        trait = RGBAColor()
        validated = trait.validate(None, None, Color(rgba=color))
        self.assertIs(
            validated, color
        )

    def test_validate_name(self):
        color = (0.4, 0.2, 0.6, 1.0)
        trait = RGBAColor()
        validated = trait.validate(None, None, "rebeccapurple")
        self.assertEqual(
            validated, color
        )

    def test_validate_hex(self):
        color = (0.4, 0.2, 0.6, 1.0)
        trait = RGBAColor()
        validated = trait.validate(None, None, "#663399ff")
        self.assertEqual(
            validated, color
        )

    def test_validate_tuple(self):
        color = (0.4, 0.2, 0.6, 0.8)
        trait = RGBAColor()
        validated = trait.validate(None, None, (0.4, 0.2, 0.6, 0.8))
        self.assertEqual(
            validated, color
        )

    def test_validate_list(self):
        color = (0.4, 0.2, 0.6, 0.8)
        trait = RGBAColor()
        validated = trait.validate(None, None, [0.4, 0.2, 0.6, 0.8])
        self.assertEqual(
            validated, color
        )

    def test_validate_rgb_list(self):
        color = (0.4, 0.2, 0.6, 1.0)
        trait = RGBAColor()
        validated = trait.validate(None, None, [0.4, 0.2, 0.6])
        self.assertEqual(
            validated, color
        )

    def test_validate_bad_string(self):
        trait = RGBAColor()
        with self.assertRaises(TraitError):
            trait.validate(None, None, "not a color")

    def test_validate_bad_object(self):
        trait = RGBAColor()
        with self.assertRaises(TraitError):
            trait.validate(None, None, object())

    def test_info(self):
        trait = RGBAColor()
        self.assertIsInstance(trait.info(), str)

    def test_default_trait(self):
        color_class = ColorClass()
        self.assertEqual(color_class.color, (1.0, 1.0, 1.0, 1.0))

    def test_set_color(self):
        color = (0.4, 0.2, 0.6, 1.0)
        color_class = ColorClass(color=Color(rgba=color))
        self.assertIs(color_class.color, color)

    def test_set_name(self):
        color = (0.4, 0.2, 0.6, 1.0)
        color_class = ColorClass(color="rebeccapurple")
        self.assertEqual(color_class.color, color)

    def test_set_hex(self):
        color = (0.4, 0.2, 0.6, 1.0)
        color_class = ColorClass(color="#663399ff")
        self.assertEqual(color_class.color, color)

    def test_set_tuple(self):
        color = (0.4, 0.2, 0.6, 1.0)
        color_class = ColorClass(color=(0.4, 0.2, 0.6, 1.0))
        self.assertEqual(color_class.color, color)

    def test_set_list(self):
        color = (0.4, 0.2, 0.6, 1.0)
        color_class = ColorClass(color=[0.4, 0.2, 0.6, 1.0])
        self.assertEqual(color_class.color, color)

    def test_set_array(self):
        color = (0.4, 0.2, 0.6, 1.0)
        color_class = ColorClass(color=np.array([0.4, 0.2, 0.6, 1.0]))
        self.assertEqual(color_class.color, color)

    def test_set_structured_dtype(self):
        color = (0.4, 0.2, 0.6, 1.0)
        arr = np.array(
            [(0.4, 0.2, 0.6, 1.0)],
            dtype=np.dtype([
                ('red', float),
                ('green', float),
                ('blue', float),
                ('alpha', float),
            ]),
        )
        color_class = ColorClass(color=arr[0])
        self.assertEqual(color_class.color, color)

    def test_get_editor(self):
        trait = RGBAColor()
        editor = trait.get_editor()

        self.assertIsInstance(editor, EditorFactory)
