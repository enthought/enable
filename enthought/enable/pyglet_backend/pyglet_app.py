"""
In other backends, we expect the UI toolkit to have a main loop running
external to Enable.

As of Pyglet 1.1, a new Pyglet App object was introduced which handles
running a main loop and efficiently dispatching to windows and event
handlers, so the former Enable PygletApp object is no longer needed,
and this file is just a stub for backwards compatibility.
"""

import pyglet

__all__ = ["get_app", "PygletApp"]

def get_app():
    """ Returns a reference to the current running Pyglet App """
    return pyglet.app

PygletApp = pyglet.app
