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

from kiva.fonttools._constants import preferred_fonts
from kiva.fonttools._score import (
    score_family, score_size, score_stretch, score_style, score_variant,
    score_weight
)


class TestFontScoring(unittest.TestCase):
    def test_score_family(self):
        # exact matches
        self.assertEqual(score_family(["Times"], "Times"), 0.0)
        closest = preferred_fonts["sans-serif"][0]
        self.assertEqual(score_family(["sans-serif"], closest), 0.0)
        self.assertEqual(score_family(["sans"], closest), 0.0)
        self.assertEqual(score_family(["unknown", "modern"], closest), 0.0)

        # fuzzy matches
        sans_count = len(preferred_fonts["sans-serif"])
        worst = preferred_fonts["sans-serif"][-1]
        expected = 0.1 * (sans_count - 1) / sans_count
        self.assertAlmostEqual(score_family(["sans-serif"], worst), expected)
        near_best = preferred_fonts["sans-serif"][1]
        expected = 0.1 * 1 / sans_count
        self.assertAlmostEqual(
            score_family(["sans-serif"], near_best), expected
        )

        # misses
        self.assertEqual(score_family(["Times"], "Arial"), 1.0)
        self.assertEqual(score_family(["serif"], "Arial"), 1.0)

    def test_score_size(self):
        # exact matches
        self.assertEqual(score_size(12.0, 12.0), 0.0)
        self.assertEqual(score_size("12.0", 12.0), 0.0)
        self.assertEqual(score_size(12.0, "12.0"), 0.0)

        # scaled exact matches
        self.assertEqual(score_size(12.0, "scalable"), 0.0)

        # fuzzy matches
        self.assertAlmostEqual(score_size(12.0, 19.2), 0.1)
        self.assertAlmostEqual(score_size(12.0, 48.0), 0.5)

        # misses
        self.assertEqual(score_size(8.0, 80.0), 1.0)
        self.assertEqual(score_size(24.0, "doesn't matter"), 1.0)

    def test_score_stretch(self):
        # exact matches
        self.assertEqual(score_stretch(500, 500), 0.0)
        self.assertEqual(score_stretch("normal", 500), 0.0)
        self.assertEqual(score_stretch(500, "normal"), 0.0)

        # fuzzy matches
        self.assertEqual(score_stretch("condensed", "semi-condensed"), 0.1)
        self.assertEqual(
            score_stretch("ultra-condensed", "ultra-expanded"), 0.8
        )

        # miss
        self.assertEqual(score_stretch(0, 1000), 1.0)

    def test_score_style(self):
        # exact matches
        self.assertEqual(score_style("italic", "italic"), 0.0)
        self.assertEqual(score_style("oblique", "oblique"), 0.0)

        # fuzzy matches
        self.assertEqual(score_style("italic", "oblique"), 0.1)
        self.assertEqual(score_style("oblique", "italic"), 0.1)

        # miss
        self.assertEqual(score_style("zany", "wacky"), 1.0)

    def test_score_variant(self):
        # exact matches
        self.assertEqual(score_variant("normal", "normal"), 0.0)
        self.assertEqual(score_variant("small-cap", "small-cap"), 0.0)

        # horseshoes and hand grenades, but not this function

        # miss
        self.assertEqual(score_variant("laden", "unladen"), 1.0)

    def test_score_weight(self):
        # exact matches
        self.assertEqual(score_weight(500, 500), 0.0)
        self.assertEqual(score_weight("500", 500), 0.0)
        self.assertEqual(score_weight(500, "500"), 0.0)
        self.assertEqual(score_weight("medium", 500), 0.0)
        self.assertEqual(score_weight(500, "medium"), 0.0)

        # fuzzy match
        self.assertEqual(score_weight(400, 500), 0.1)

        # miss
        self.assertEqual(score_weight(0, 1000), 1.0)
