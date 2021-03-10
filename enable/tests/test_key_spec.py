# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import collections
import unittest

from enable.api import KeySpec

Event = collections.namedtuple(
    "Event", "character,alt_down,control_down,shift_down"
)


class TestKeySpec(unittest.TestCase):
    def test_basics(self):
        spec = KeySpec("Right", "control", ignore=["shift"])

        self.assertEqual(spec.key, "Right")
        self.assertTrue(spec.control)
        self.assertSetEqual(spec.ignore, {"shift"})
        self.assertFalse(spec.alt)
        self.assertFalse(spec.shift)

        event = Event("k", False, True, False)
        self.assertFalse(spec.match(event))

        event = Event("Right", False, True, False)
        self.assertTrue(spec.match(event))

    def test_from_string(self):
        spec = KeySpec.from_string("Shift+Control+z")
        self.assertSetEqual(spec.ignore, {"alt"})

        event = Event("z", False, True, True)
        self.assertTrue(spec.match(event))
