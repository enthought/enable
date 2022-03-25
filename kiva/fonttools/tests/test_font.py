# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Tests for kiva.fonttools.font
"""
import os
import unittest

from kiva.api import (
    BOLD, BOLD_ITALIC, Font, ITALIC, MODERN, NORMAL, ROMAN, WEIGHT_BOLD,
    WEIGHT_LIGHT
)
from kiva.fonttools import str_to_font
from kiva.fonttools.tests._testing import patch_global_font_manager


class TestFont(unittest.TestCase):
    def setUp(self):
        # Invalidate the global font manager cache to avoid test interaction
        # as well as catching erroneous assumption on an existing cache.
        font_manager_patcher = patch_global_font_manager(None)
        font_manager_patcher.start()
        self.addCleanup(font_manager_patcher.stop)

    def test_find_font_empty_name(self):
        # This test relies on the fact there exists some fonts on the system
        # that the font manager can load. Ideally we should be able to redirect
        # the path from which the font manager loads font files, then this test
        # can be less fragile.
        font = Font(face_name="")
        spec = font.findfont()
        self.assertTrue(os.path.exists(spec.filename))

    def test_find_font_some_face_name(self):
        font = Font(face_name="ProbablyNotFound")

        # There will be warnings as there will be no match for the requested
        # face name.
        with self.assertWarns(UserWarning):
            spec = font.findfont()
        self.assertTrue(os.path.exists(spec.filename))

    def test_find_font_name(self):
        font = Font(face_name="ProbablyNotFound")

        # There will be warnings as there will be no match for the requested
        # face name.
        with self.assertWarns(UserWarning):
            name = font.findfontname()

        # Name should be nonempty.
        self.assertGreater(len(name), 0)

    def test_find_font_for_language(self):
        font = Font(face_name="")

        # Nearly every font supports Latin script, so this shouldn't fail
        spec = font.findfont(language="Latin")
        self.assertTrue(os.path.exists(spec.filename))

        # There will be warnings for an unknown language
        with self.assertWarns(UserWarning):
            spec = font.findfont(language="FancyTalk")
        self.assertTrue(os.path.exists(spec.filename))

    def test_str_to_font(self):
        # Simple
        from_str = str_to_font("modern 10")
        from_ctor = Font(family=MODERN, size=10)
        self.assertEqual(from_ctor, from_str)

        # Some complexity
        from_str = str_to_font("roman bold italic 12")
        from_ctor = Font(family=ROMAN, weight=WEIGHT_BOLD, style=ITALIC, size=12)
        self.assertEqual(from_ctor, from_str)

        # Lots of complexity
        from_str = str_to_font("Times roman bold italic underline 72")
        from_ctor = Font(
            "Times",
            family=ROMAN,
            weight=WEIGHT_BOLD,
            style=ITALIC,
            size=72,
            underline=1,
        )
        self.assertEqual(from_ctor, from_str)

    def test_is_bold_false(self):
        for weight in range(100, 501, 100):
            with self.subTest(weight=weight):
                font = Font(weight=weight)

                self.assertFalse(font.is_bold())

    def test_is_bold_true(self):
        for weight in range(600, 1001, 100):
            with self.subTest(weight=weight):
                font = Font(weight=weight)

                self.assertTrue(font.is_bold())

    def test_weight_warnings(self):
        # Don't use BOLD as a weight
        with self.assertWarns(DeprecationWarning):
            font = Font(weight=BOLD)
        self.assertEqual(font.weight, WEIGHT_BOLD)

        # Don't use BOLD as a style
        with self.assertWarns(DeprecationWarning):
            font = Font(style=BOLD)
        self.assertEqual(font.weight, WEIGHT_BOLD)
        self.assertEqual(font.style, NORMAL)

        # Don't use BOLD_ITALIC as a style
        with self.assertWarns(DeprecationWarning):
            font = Font(style=BOLD_ITALIC)
        self.assertEqual(font.weight, WEIGHT_BOLD)
        self.assertEqual(font.style, ITALIC)

        # Ignore BOLD style if weight is not normal
        with self.assertWarns(DeprecationWarning):
            font = Font(style=BOLD, weight=WEIGHT_LIGHT)
        self.assertEqual(font.weight, WEIGHT_LIGHT)
        self.assertEqual(font.style, NORMAL)

        with self.assertWarns(DeprecationWarning):
            font = Font(style=BOLD_ITALIC, weight=WEIGHT_LIGHT)
        self.assertEqual(font.weight, WEIGHT_LIGHT)
        self.assertEqual(font.style, ITALIC)

    def test_font_query_warnings(self):
        # Don't use BOLD as a weight
        font = Font()
        font.weight = BOLD
        with self.assertWarns(DeprecationWarning):
            query = font.findfont()
        self.assertEqual(query.weight, WEIGHT_BOLD)

        # Don't use BOLD as a style
        font = Font()
        font.style = BOLD
        with self.assertWarns(DeprecationWarning):
            query = font.findfont()
        self.assertEqual(query.weight, WEIGHT_BOLD)
        self.assertEqual(query.style, "normal")

        # Don't use BOLD_ITALIC as a style
        font = Font()
        font.style = BOLD_ITALIC
        with self.assertWarns(DeprecationWarning):
            query = font.findfont()
        self.assertEqual(query.weight, WEIGHT_BOLD)
        self.assertEqual(query.style, "italic")

        # Ignore BOLD style if weight is not normal
        font = Font()
        font.weight = WEIGHT_LIGHT
        font.style = BOLD
        with self.assertWarns(DeprecationWarning):
            query = font.findfont()
        self.assertEqual(query.weight, WEIGHT_LIGHT)
        self.assertEqual(query.style, "normal")

        font = Font()
        font.weight = WEIGHT_LIGHT
        font.style = BOLD_ITALIC
        with self.assertWarns(DeprecationWarning):
            query = font.findfont()
        self.assertEqual(query.weight, WEIGHT_LIGHT)
        self.assertEqual(query.style, "italic")
