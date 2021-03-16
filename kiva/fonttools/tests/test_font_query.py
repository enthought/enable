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

from kiva.fonttools._query import FontQuery


class TestFontQuery(unittest.TestCase):
    def setUp(self):
        self.query = FontQuery(
            family=["serif"],
            style="italic",
            variant="small-caps",
            weight="bold",
            stretch="ultra-condensed",
            size=18,
        )

    def test_copying(self):
        query_copy = self.query.copy()

        for attr in ("_family", "_slant", "_variant", "_weight",
                     "_stretch", "_size", "_file"):
            self.assertEqual(getattr(self.query, attr),
                             getattr(query_copy, attr))

        # Compare the strings too
        self.assertEqual(str(self.query), str(query_copy))
        # And the hashes
        self.assertEqual(hash(self.query), hash(query_copy))

    def test_getters(self):
        self.assertListEqual(self.query.get_family(), ["serif"])
        self.assertIsNone(self.query.get_file())
        self.assertEqual(self.query.get_size(), 18)
        self.assertEqual(self.query.get_slant(), "italic")
        self.assertEqual(self.query.get_stretch(), "ultra-condensed")
        self.assertEqual(self.query.get_style(), "italic")
        self.assertEqual(self.query.get_variant(), "small-caps")
        self.assertEqual(self.query.get_weight(), "bold")

    def test_setters(self):
        # Family is always converted to a list
        self.query.set_family("cursive")
        self.assertListEqual(self.query.get_family(), ["cursive"])
        # Family can be unset
        self.query.set_family(None)
        self.assertIsNone(self.query.get_family())
        # Family can be a bytestring
        self.query.set_family("Arial".encode("utf8"))
        self.assertListEqual(self.query.get_family(), ["Arial"])

        filename = "not a real file.ttf"
        self.query.set_file(filename)
        self.assertEqual(self.query.get_file(), filename)

        # set_name is a synonym for set_family
        self.query.set_name("Verdana")
        self.assertListEqual(self.query.get_family(), ["Verdana"])

        # set_style has requirements
        with self.assertRaises(ValueError):
            self.query.set_style("post-modern")

        self.query.set_style("oblique")
        self.assertEqual(self.query.get_style(), "oblique")

        # set_variant has requirements
        with self.assertRaises(ValueError):
            self.query.set_variant("rad")

        self.query.set_variant("normal")
        self.assertEqual(self.query.get_variant(), "normal")

        # set_weight takes many input types
        with self.assertRaises(ValueError):
            self.query.set_weight("superduperdark")
        with self.assertRaises(ValueError):
            self.query.set_weight(-42)
        with self.assertRaises(ValueError):
            self.query.set_weight(3000)

        self.query.set_weight(500)
        self.assertEqual(self.query.get_weight(), 500)
        self.query.set_weight("bold")
        self.assertEqual(self.query.get_weight(), "bold")

        # set_stretch has requirements
        with self.assertRaises(ValueError):
            self.query.set_stretch("condensed-matter")
        with self.assertRaises(ValueError):
            self.query.set_stretch(-42)
        with self.assertRaises(ValueError):
            self.query.set_stretch(3000)

        self.query.set_stretch("semi-condensed")
        self.assertEqual(self.query.get_stretch(), "semi-condensed")
        self.query.set_stretch(300)
        self.assertEqual(self.query.get_stretch(), 300)
        # default
        self.query.set_stretch(None)
        self.assertEqual(self.query.get_stretch(), 500)

        # set_size has requirements
        with self.assertRaises(ValueError):
            self.query.set_size("itsy-bitsy")

        self.query.set_size(36)
        self.assertEqual(self.query.get_size(), 36.0)
