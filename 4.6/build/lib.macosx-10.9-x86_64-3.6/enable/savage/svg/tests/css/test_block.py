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

from enable.savage.svg.css import block


class TestBlockParsing(unittest.TestCase):
    def testBlock(self):
        """ Not a valid CSS statement, but a valid block
            This tests some abuses of quoting and escaping
            See http://www.w3.org/TR/REC-CSS2/syndata.html Section 4.1.6
        """
        self.assertEqual(
            [["causta:", '"}"', "+", "(", ["7"], "*", "'\\''", ")"]],
            block.block.parseString(
                r"""{ causta: "}" + ({7} * '\'') }"""
            ).asList(),
        )
