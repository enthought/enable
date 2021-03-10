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

from kiva.fonttools._font_properties import FontProperties


class TestFontProperties(unittest.TestCase):
    def setUp(self):
        self.fp = FontProperties(
            family=["serif"],
            style="italic",
            variant="small-caps",
            weight="bold",
            stretch="ultra-condensed",
            size=18,
        )

    def test_copying(self):
        fp_copy = self.fp.copy()

        for attr in ("_family", "_slant", "_variant", "_weight",
                     "_stretch", "_size", "_file"):
            self.assertEqual(getattr(self.fp, attr), getattr(fp_copy, attr))

        # Compare the strings too
        self.assertEqual(str(self.fp), str(fp_copy))
        # And the hashes
        self.assertEqual(hash(self.fp), hash(fp_copy))

    def test_getters(self):
        self.assertListEqual(self.fp.get_family(), ["serif"])
        self.assertIsNone(self.fp.get_file())
        self.assertEqual(self.fp.get_size(), 18)
        self.assertEqual(self.fp.get_slant(), "italic")
        self.assertEqual(self.fp.get_stretch(), "ultra-condensed")
        self.assertEqual(self.fp.get_style(), "italic")
        self.assertEqual(self.fp.get_variant(), "small-caps")
        self.assertEqual(self.fp.get_weight(), "bold")

    def test_setters(self):
        # Family is always converted to a list
        self.fp.set_family("cursive")
        self.assertListEqual(self.fp.get_family(), ["cursive"])
        # Family can be unset
        self.fp.set_family(None)
        self.assertIsNone(self.fp.get_family())
        # Family can be a bytestring
        self.fp.set_family("Arial".encode("utf8"))
        self.assertListEqual(self.fp.get_family(), ["Arial"])

        filename = "not a real file.ttf"
        self.fp.set_file(filename)
        self.assertEqual(self.fp.get_file(), filename)

        # set_name is a synonym for set_family
        self.fp.set_name("Verdana")
        self.assertListEqual(self.fp.get_family(), ["Verdana"])

        # set_style has requirements
        with self.assertRaises(ValueError):
            self.fp.set_style("post-modern")

        self.fp.set_style("oblique")
        self.assertEqual(self.fp.get_style(), "oblique")

        # set_variant has requirements
        with self.assertRaises(ValueError):
            self.fp.set_variant("rad")

        self.fp.set_variant("normal")
        self.assertEqual(self.fp.get_variant(), "normal")

        # set_weight takes many input types
        with self.assertRaises(ValueError):
            self.fp.set_weight("superduperdark")
        with self.assertRaises(ValueError):
            self.fp.set_weight(-42)
        with self.assertRaises(ValueError):
            self.fp.set_weight(3000)

        self.fp.set_weight(500)
        self.assertEqual(self.fp.get_weight(), 500)
        self.fp.set_weight("bold")
        self.assertEqual(self.fp.get_weight(), "bold")

        # set_stretch has requirements
        with self.assertRaises(ValueError):
            self.fp.set_stretch("condensed-matter")
        with self.assertRaises(ValueError):
            self.fp.set_stretch(-42)
        with self.assertRaises(ValueError):
            self.fp.set_stretch(3000)

        self.fp.set_stretch("semi-condensed")
        self.assertEqual(self.fp.get_stretch(), "semi-condensed")
        self.fp.set_stretch(300)
        self.assertEqual(self.fp.get_stretch(), 300)
        # default
        self.fp.set_stretch(None)
        self.assertEqual(self.fp.get_stretch(), 500)

        # set_size has requirements
        with self.assertRaises(ValueError):
            self.fp.set_size("itsy-bitsy")

        self.fp.set_size("medium")
        self.assertEqual(self.fp.get_size(), "medium")
        self.fp.set_size(36)
        self.assertEqual(self.fp.get_size(), 36.0)
