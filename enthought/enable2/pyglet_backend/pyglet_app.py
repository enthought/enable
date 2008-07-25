# proxy

__all__ = ["get_app", "PygletApp"]

from enthought.enable.pyglet_backend.pyglet_app import *


# Import the objects which are not declared in __all__,
# but are still defined in the real module, such that people
# can import them explicitly when needed, just as they could
# with the real module.
#
# It is unlikely that someone will import these objects, since
# they start with '_'.  However, the proxy's job is to mimic the
# behavior of the real module as closely as possible.
# The proxy's job is not to define or change the API.
#
from enthought.enable.pyglet_backend.pyglet_app import _CurrentApp, _PygletApp

