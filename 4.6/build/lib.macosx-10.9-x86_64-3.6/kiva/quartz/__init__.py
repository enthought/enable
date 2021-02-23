# (C) Copyright 2004-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import sys

if sys.platform == "darwin":
    from kiva.quartz.mac_context import get_mac_context  # noqa: F401

    def get_macport(dc):
        """ Returns the Port or the CGContext of a wxDC (or child class)
        instance.
        """
        if "GetCGContext" in dir(dc):
            ptr = dc.GetCGContext()
            return int(ptr)
        else:
            from macport import get_macport as _get_macport

            return _get_macport(str(dc.this))
