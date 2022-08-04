# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
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
from itertools import chain, combinations
import os
import unittest

from kiva.constants import (
    BOLD, BOLD_ITALIC, DECORATIVE, DEFAULT, ITALIC, MODERN, NORMAL, ROMAN,
    SCRIPT, TELETYPE, WEIGHT_BOLD, WEIGHT_LIGHT, WEIGHT_NORMAL, SWISS,
)
from kiva.fonttools._constants import font_family_aliases, preferred_fonts
from kiva.fonttools.tests._testing import patch_global_font_manager
from kiva.fonttools.font import (
    DECORATIONS, FAMILIES, NOISE, STYLES, WEIGHTS, Font, str_to_font,
    simple_parser,
)


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

        # Using extra font weights
        from_str = str_to_font("Times roman light italic underline 72")
        from_ctor = Font(
            "Times",
            family=ROMAN,
            weight=WEIGHT_LIGHT,
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
            query = font._make_font_query()
        self.assertEqual(query.get_weight(), WEIGHT_BOLD)

        # Don't use BOLD as a style
        font = Font()
        font.style = BOLD
        with self.assertWarns(DeprecationWarning):
            query = font._make_font_query()
        self.assertEqual(query.get_weight(), WEIGHT_BOLD)
        self.assertEqual(query.get_style(), "normal")

        # Don't use BOLD_ITALIC as a style
        font = Font()
        font.style = BOLD_ITALIC
        with self.assertWarns(DeprecationWarning):
            query = font._make_font_query()
        self.assertEqual(query.get_weight(), WEIGHT_BOLD)
        self.assertEqual(query.get_style(), "italic")

        # Ignore BOLD style if weight is not normal
        font = Font()
        font.weight = WEIGHT_LIGHT
        font.style = BOLD
        with self.assertWarns(DeprecationWarning):
            query = font._make_font_query()
        self.assertEqual(query.get_weight(), WEIGHT_LIGHT)
        self.assertEqual(query.get_style(), "normal")

        font = Font()
        font.weight = WEIGHT_LIGHT
        font.style = BOLD_ITALIC
        with self.assertWarns(DeprecationWarning):
            query = font._make_font_query()
        self.assertEqual(query.get_weight(), WEIGHT_LIGHT)
        self.assertEqual(query.get_style(), "italic")

    def test_family_queries(self):
        # regression test for Enable #971
        # this ensures every font family creates a valid query populated
        # with query families that work
        families = [
            DECORATIVE, DEFAULT, MODERN, ROMAN, SCRIPT, SWISS, TELETYPE
        ]
        for family in families:
            with self.subTest(family=family):
                font = Font(family=family)
                query = font._make_font_query()
                query_family = query.get_family()[0]

                self.assertEqual(query_family, Font.familymap.get(family))
                self.assertIn(query_family, font_family_aliases)
                self.assertIn(query_family, preferred_fonts)


class TestSimpleParser(unittest.TestCase):

    def test_empty(self):
        properties = simple_parser("")
        self.assertEqual(
            properties,
            {
                'face_name': "",
                'family': DEFAULT,
                'size': 10,
                'weight': WEIGHT_NORMAL,
                'style': NORMAL,
                'underline': False,
            },
        )

    def test_typical(self):
        properties = simple_parser(
            "10 pt bold italic underline Helvetica sans-serif")
        self.assertEqual(
            properties,
            {
                'face_name': "Helvetica",
                'family': SWISS,
                'size': 10,
                'weight': WEIGHT_BOLD,
                'style': ITALIC,
                'underline': True,
            },
        )

    def test_noise(self):
        for noise in NOISE:
            with self.subTest(noise=noise):
                properties = simple_parser(noise)
                self.assertEqual(
                    properties,
                    {
                        'face_name': "",
                        'family': DEFAULT,
                        'size': 10,
                        'weight': WEIGHT_NORMAL,
                        'style': NORMAL,
                        'underline': False,
                    },
                )

    def test_generic_families(self):
        for family, constant in FAMILIES.items():
            with self.subTest(family=family):
                properties = simple_parser(family)
                self.assertEqual(
                    properties,
                    {
                        'face_name': "",
                        'family': constant,
                        'size': 10,
                        'weight': WEIGHT_NORMAL,
                        'style': NORMAL,
                        'underline': False,
                    },
                )

    def test_size(self):
        for size in [12, 24]:
            with self.subTest(size=size):
                properties = simple_parser(str(size))
                self.assertEqual(
                    properties,
                    {
                        'face_name': "",
                        'family': DEFAULT,
                        'size': size,
                        'weight': WEIGHT_NORMAL,
                        'style': NORMAL,
                        'underline': False,
                    },
                )

    def test_weight(self):
        for weight, constant in WEIGHTS.items():
            with self.subTest(weight=weight):
                properties = simple_parser(weight)
                self.assertEqual(
                    properties,
                    {
                        'face_name': "",
                        'family': DEFAULT,
                        'size': 10,
                        'weight': constant,
                        'style': NORMAL,
                        'underline': False,
                    },
                )

    def test_style(self):
        for style, constant in STYLES.items():
            with self.subTest(style=style):
                properties = simple_parser(style)
                self.assertEqual(
                    properties,
                    {
                        'face_name': "",
                        'family': DEFAULT,
                        'size': 10,
                        'weight': WEIGHT_NORMAL,
                        'style': constant,
                        'underline': False,
                    },
                )

    def test_decorations(self):
        # get powerset iterator of DECORATIONS
        all_decorations = chain.from_iterable(
            combinations(DECORATIONS, n)
            for n in range(len(DECORATIONS) + 1)
        )
        for decorations in all_decorations:
            with self.subTest(decorations=decorations):
                properties = simple_parser(" ".join(decorations))
                self.assertEqual(
                    properties,
                    {
                        'face_name': "",
                        'family': DEFAULT,
                        'size': 10,
                        'weight': WEIGHT_NORMAL,
                        'style': NORMAL,
                        'underline': 'underline' in decorations,
                    },
                )
