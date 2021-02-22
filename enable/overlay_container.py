# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from .container import Container
from .simple_layout import (
    simple_container_get_preferred_size,
    simple_container_do_layout,
)


class OverlayContainer(Container):
    """ A container that stretches all its components to fit within its space.
    All of its components must therefore be resizable.
    """

    def get_preferred_size(self, components=None):
        """ Returns the size (width,height) that is preferred for this
        component.

        Overrides PlotComponent
        """
        return simple_container_get_preferred_size(self, components=components)

    def _do_layout(self):
        """ Actually performs a layout (called by do_layout()).
        """
        simple_container_do_layout(self)
