# (C) Copyright 2008-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

import unittest

from traits.api import HasTraits, TraitError

from kiva.fonttools.font import Font
from kiva import constants
from ..kiva_font_trait import (
    KivaFont, TraitKivaFont, font_families, font_styles, font_weights
)


class FontExample(HasTraits):

    font = KivaFont()


class TestKivaFont(unittest.TestCase):

    def test_validate_str(self):
        expected_outcomes = {}
        expected_outcomes[""] = Font(size=10, family=constants.DEFAULT)

        for weight, kiva_weight in font_weights.items():
            expected_outcomes[weight] = Font(weight=kiva_weight, size=10, family=constants.DEFAULT)

        for style, kiva_style in font_styles.items():
            expected_outcomes[style] = Font(style=kiva_style, size=10, family=constants.DEFAULT)

        expected_outcomes["underline"] = Font(underline=True, size=10, family=constants.DEFAULT)

        expected_outcomes["18"] = Font(size=18, family=constants.DEFAULT)
        expected_outcomes["18 pt"] = Font(size=18, family=constants.DEFAULT)
        expected_outcomes["18 point"] = Font(size=18, family=constants.DEFAULT)

        for family, kiva_family in font_families.items():
            expected_outcomes[family] = Font(family=kiva_family, size=10)

        expected_outcomes["Courier"] = Font("Courier", size=10, family=constants.DEFAULT)
        expected_outcomes["Comic Sans"] = Font("Comic Sans", size=10, family=constants.DEFAULT)
        expected_outcomes["18 pt Bold Italic Underline Comic Sans script"] = Font(
            "Comic Sans", 18, constants.SCRIPT, weight=constants.WEIGHT_BOLD,
            style=constants.ITALIC, underline=True,
        )

        for name, expected in expected_outcomes.items():
            with self.subTest(name=name):
                example = FontExample(font=name)
                result = example.font

                # test we get expected font
                self.assertIsInstance(result, Font)
                self.assertEqual(result, expected)

    def test_validate_font(self):
        font = Font("Comic Sans", 18)
        example = FontExample(font=font)

        result = example.font

        # test we get expected font
        self.assertIsInstance(result, Font)
        self.assertIs(result, font)

    def test_font_trait_default(self):
        example = FontExample()

        self.assertIsInstance(example.font, Font)
        self.assertEqual(example.font, Font(size=12, family=constants.MODERN))

    def test_font_trait_none(self):
        with self.assertRaises(TraitError):
            FontExample(font=None)
