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

from traits.api import HasStrictTraits, TraitError, Undefined

from enable.enable_traits import ScrollBarRange, ScrollPosition


class DummyScrollBar(HasStrictTraits):

    range = ScrollBarRange()

    position = ScrollPosition()


class TestScrollBarTraits(unittest.TestCase):

    def test_range_default(self):
        scroll_bar = DummyScrollBar()
        self.assertIs(scroll_bar.range, Undefined)

    def test_position_default(self):
        scroll_bar = DummyScrollBar()
        self.assertEqual(scroll_bar.position, 0.0)

    def test_range_set(self):
        scroll_bar = DummyScrollBar()
        values = [
            # standard
            ((0.0, 100.0, 10.0, 1.0), (0.0, 100.0, 10.0, 1.0)),
            # standard, but ints
            ((0, 100, 10, 1), (0.0, 100.0, 10.0, 1.0)),
            # standard, but list
            ([0.0, 100.0, 10.0, 1.0], (0.0, 100.0, 10.0, 1.0)),
            # negative low
            ((-100.0, 100.0, 10.0, 1.0), (-100.0, 100.0, 10.0, 1.0)),
            # high/low reversed
            ((100.0, 0.0, 10.0, 1.0), (0.0, 100.0, 10.0, 1.0)),
            # high/low equal
            ((0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
            # page size greater than range
            ((0.0, 100.0, 1000.0, 1.0), (0.0, 100.0, 100.0, 1.0)),
            # page size negative
            ((0.0, 100.0, -10.0, 1.0), (0.0, 100.0, 0.0, 0.0)),
            # line size greater than page size
            ((0.0, 100.0, 10.0, 100.0), (0.0, 100.0, 10.0, 10.0)),
            # line size negative
            ((0.0, 100.0, 10.0, -1.0), (0.0, 100.0, 10.0, 0.0)),
        ]
        for value, expected in values:
            with self.subTest(value=value):
                scroll_bar.range = value
                self.assertIsInstance(scroll_bar.range, tuple)
                self.assertTrue(
                    all(isinstance(x, float) for x in scroll_bar.range)
                )
                self.assertEqual(scroll_bar.range, expected)

    def test_range_set_invalid(self):
        scroll_bar = DummyScrollBar()
        values = [
            # too short
            (0.0, 100.0, 10.0),
            # too long
            (0.0, 100.0, 10.0, 1.0, 1.0),
            # not floats
            ('0.0', '100.0', '10.0', '1.0'),
            # not comparable
            (None, None, None, None),
            # None not allowed
            None,
            # sets not OK
            {0, 100, 10, 1},
        ]
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(TraitError):
                    scroll_bar.range = value

    def test_position_set(self):
        scroll_bar = DummyScrollBar(range=(0.0, 100.0, 10.0, 1.0))
        values = [
            # typical
            (10.0, 10.0),
            # typical, but an int
            (10, 10.0),
            # at top
            (0.0, 0.0),
            # at bottom
            (90.0, 90.0),
            # above top
            (-10.0, 0.0),
            # below bottom
            (100.0, 90.0),
        ]
        for value, expected in values:
            with self.subTest(value=value):
                scroll_bar.position = value
                self.assertIsInstance(scroll_bar.position, float)
                self.assertEqual(scroll_bar.position, expected)
