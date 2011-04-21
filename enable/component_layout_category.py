""" FIXME:
Tentative implementation of a new layout mechanism. Unused and unworking.
"""


# Enthought library imports
from traits.api import Any, Category, Enum

# Singleton representing the default Enable layout manager
DefaultLayoutController = LayoutController()


class ComponentLayoutCategory(Category, Component):

    """ Properties defining how a component should be laid out. """

    resizable = Enum('h', 'v')

    max_width = Any

    min_width = Any

    max_height = Any

    min_height = Any

    # Various alignment and positioning functions

    def set_padding(self, left, right, top, bottom):
        pass

    def get_padding(self):
        pass
