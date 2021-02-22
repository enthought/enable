# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
# -----------------------------------------------------------------------------
# Copyright (c) 2007, Riverbank Computing Limited
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# -----------------------------------------------------------------------------
from pyface.base_toolkit import find_toolkit
from traits.etsconfig.api import ETSConfig


def _init_toolkit():
    """ Initialise the current toolkit.
    """
    if not ETSConfig.toolkit:
        # Force Traits to decide on its toolkit if it hasn't already
        from traitsui.api import toolkit as traits_toolkit

        traits_toolkit()


# Do this once then disappear.
_init_toolkit()
del _init_toolkit

# The toolkit_object function.
toolkit_object = find_toolkit("enable.toolkits")
