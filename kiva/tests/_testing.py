# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" This module contains utilities for testing within Kiva.

This is not a public module and should not to be used outside of Kiva.
"""

import unittest

from traits.etsconfig.api import ETSConfig


def is_wx():
    """ Return true if the toolkit backend is wx. """
    return ETSConfig.toolkit == "wx"


skip_if_not_wx = unittest.skipIf(not is_wx(), "Test only for wx")
