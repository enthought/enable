#------------------------------------------------------------------------------
# Copyright (c) 2007, Riverbank Computing Limited
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
#------------------------------------------------------------------------------

# Standard library imports.
import sys

# Enthought library imports.
from enthought.etsconfig.api import ETSConfig


# This is set to the api module path for the selected backend.
_toolkit_backend = None


def _init_toolkit():
    """ Initialise the current toolkit. """

    # If not defined, use the same toolkit as Traits UI
    if not ETSConfig.enable_toolkit:

        # Force Traits to decide on its toolkit if it hasn't already
        from enthought.traits.ui.toolkit import toolkit as traits_toolkit
        traits_toolkit()

        ETSConfig.enable_toolkit = ETSConfig.toolkit

    # Import the selected backend
    backend = 'enthought.enable.%s_backend.api' % ETSConfig.enable_toolkit
    try:
        __import__(backend)
    except ImportError, SystemExit:
        raise ImportError, "Unable to import an Enable backend for the %s " \
            "toolkit." % ETSConfig.enable_toolkit

    # Save the imported toolkit module.
    global _toolkit_backend
    _toolkit_backend = backend

# Do this once then disappear.
_init_toolkit()
del _init_toolkit


def toolkit_object(name):
    """ Return the toolkit specific object with the given name. """

    try:
        be_obj = getattr(sys.modules[_toolkit_backend], name)
    except AttributeError:
        raise NotImplementedError("the %s enable backend doesn't implement %s" % (ETSConfig.toolkit, name))

    return be_obj
