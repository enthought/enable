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
from traceback import format_exception_only

# Enthought library imports.
from traits.etsconfig.api import ETSConfig


# This is set to the api module path for the selected backend.
_toolkit_backend = None


def _init_toolkit():
    """ Initialise the current toolkit. """

    if not ETSConfig.toolkit:
        # Force Traits to decide on its toolkit if it hasn't already
        from traitsui.toolkit import toolkit as traits_toolkit
        traits_toolkit()

    # Import the selected backend
    backend = 'enable.%s.%s' % (ETSConfig.toolkit, ETSConfig.kiva_backend)
    try:
        __import__(backend)
    except (ImportError, SystemExit):
        t, v, _tb = sys.exc_info()
        raise ImportError, "Unable to import the %s backend for the %s " \
                "toolkit (reason: %s)." % (ETSConfig.kiva_backend, ETSConfig.toolkit,
                        format_exception_only(t, v))

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
        raise NotImplementedError("the %s.%s enable backend doesn't implement %s" %
                                  (ETSConfig.toolkit, ETSConfig.kiva_backend, name))

    return be_obj
