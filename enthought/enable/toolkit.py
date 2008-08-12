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

    # Toolkits to check for if none is explicitly specified.
    known_toolkits = ('wx', 'qt4', 'pyglet')

    # Get the toolkit.
    toolkit = ETSConfig.enable_toolkit

    if toolkit:
        toolkits = (toolkit, )
    else:
        toolkits = known_toolkits

    for tk in toolkits:
        # Try and import the toolkit's enable backend.
        be = 'enthought.enable.%s_backend.api' % tk

        try:
            __import__(be)
            break
        except ImportError:
            pass
    else:
        if toolkit:
            raise ImportError, "unable to import an enable backend for the %s toolkit" % toolkit
        else:
            raise ImportError, "unable to import an enable backend for any of the %s toolkits" % ", ".join(known_toolkits)

    # In case we have just decided on a toolkit, tell everybody else.
    ETSConfig.enable_toolkit = tk

    # Save the imported toolkit module.
    global _toolkit_backend
    _toolkit_backend = be


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
