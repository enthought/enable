# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# Standard library imports
import sys

# ETS imports
from traits.etsconfig.api import ETSConfig


def _init_toolkit():
    """ Initialise the current toolkit.
    """

    # Force Traits to decide on its toolkit if it hasn't already
    from traitsui.api import toolkit as traits_toolkit

    traits_toolkit()

    # Import the selected backend
    backend = "enable.savage.trait_defs.ui.%s" % ETSConfig.toolkit
    try:
        __import__(backend)
    except (ImportError, SystemExit):
        raise ImportError(
            "Unable to import a Savage backend for the %s "
            "toolkit." % ETSConfig.toolkit
        )

    # Save the imported toolkit module.
    global _toolkit_backend
    _toolkit_backend = backend


# Do this once then disappear.
_init_toolkit()
del _init_toolkit


def toolkit_object(name, raise_exceptions=False):
    """ Return the toolkit specific object with the given name.  The name
        consists of the relative module path and the object name separated by a
        colon.
    """

    mname, oname = name.split(":")

    class Unimplemented(object):
        """ This is returned if an object isn't implemented by the selected
            toolkit. It raises an exception if it is ever instantiated.
        """

        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "The %s Savage backend doesn't implement"
                "%s" % (ETSConfig.toolkit, oname)
            )

    be_obj = Unimplemented
    be_mname = _toolkit_backend + "." + mname
    try:
        __import__(be_mname)
        try:
            be_obj = getattr(sys.modules[be_mname], oname)
        except AttributeError as e:
            if raise_exceptions:
                raise e
    except ImportError as e:
        if raise_exceptions:
            raise e

    return be_obj
