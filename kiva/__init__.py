#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# some parts copyright 2002 by Space Telescope Science Institute
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#------------------------------------------------------------------------------

"""
A multi-platform DisplayPDF vector drawing engine.
Part of the Enable project of the Enthought Tool Suite.
"""

from kiva._version import full_version as __version__

from .constants import *
from .fonttools import Font

import os
if os.environ.has_key('KIVA_WISHLIST'):
    from warnings import warn
    warn("Use of the KIVA_WISHLIST environment variable to select Kiva backends"
         "is no longer supported.")
del os
