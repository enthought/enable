# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""
Example Application Support
===========================

This module is meant for internal use only and it is not meant for use in
library code. Importing from this module is deprecated and it will be removed
in Enable 6.0. We highly recommend that you update your code and vendorize what
is necessary.

"""
import warnings

from enable.examples._example_application import DemoApplication, demo_main

warnings.warn(
    "This module is meant for internal use only and it is not meant for use in"
    " library code. Importing from this module is deprecated and it will be"
    " removed in Enable 6.0. We highly recommend that you update your code and"
    " vendorize what is necessary.",
    DeprecationWarning,
    stacklevel=2
)
