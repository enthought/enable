# (C) Copyright 2005-2023 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import numpy as np
from numpy import all, ravel


class Utils(object):
    def assertRavelEqual(self, x, y):
        self.assertTrue(
            (ravel(x) == ravel(y)).all(), "\n%s\n !=\n%s" % (x, y)
        )
