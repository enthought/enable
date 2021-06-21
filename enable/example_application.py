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

This module provides a simple Pyface application that can be used by examples
in places where a DemoFrame is insufficient. Note this has been moved to
sit in enable/examples.  This module is kept for backwards compatibility.

"""
import warnings

from enable.examples._example_application import DemoApplication, demo_main

warnings.warn(
    "Importing from this module is deprecated, and this module will be"
    " removed in a future release.",
    DeprecationWarning,
    stacklevel=2
)
