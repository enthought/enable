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

from kiva.agg import GraphicsContextSystem


class GraphicsContextSystemTestCase(unittest.TestCase):
    def test_creation(self):
        """ Simply create and destroy multiple objects.  This silly
            test crashed when we transitioned from Numeric 23.1 to 23.8.
            That problem is fixed now.
        """
        for i in range(10):
            gc = GraphicsContextSystem((100, 100), "rgba32")
            del gc
