# (C) Copyright 2025 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Compatibility workaround module

Module to workaround compatibility issues with different versions of
the enable dependencies.

"""

try:
    from numpy import sometrue
except ImportError:
    from numpy import any as sometrue
