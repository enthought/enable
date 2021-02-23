# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Abstract base class for overlays.

This class is primarily used so that tools can easily distinguish between
items underneath them.
"""

from traits.api import Instance

from .component import Component


class AbstractOverlay(Component):
    """ The base class for overlays and underlays of the area.

    The only default additional feature of an overlay is that it implements
    an overlay() drawing method that overlays this component on top of
    another, without the components necessarily having an object
    containment-ownership relationship.
    """

    # The component that this object overlays. This can be None. By default, if
    # this object is called to draw(), it tries to render onto this component.
    component = Instance(Component)

    # The default layer that this component draws into.
    draw_layer = "overlay"

    # The background color (overrides PlotComponent).
    # Typically, an overlay does not render a background.
    bgcolor = "transparent"

    # ----------------------------------------------------------------------
    # Abstract methods (to be implemented by subclasses)
    # ----------------------------------------------------------------------

    def overlay(self, other_component, gc, view_bounds=None, mode="normal"):
        """ Draws this component overlaid on another component.
        """
        # Subclasses should implement this method.
        pass

    def _do_layout(self, component=None):
        """ Called by do_layout() to do an actual layout call; it bypasses some
        additional logic to handle null bounds and setting **_layout_needed**.
        """
        pass

    # ----------------------------------------------------------------------
    # Concrete methods / reimplementations of Component methods
    # ----------------------------------------------------------------------

    def __init__(self, component=None, *args, **kw):
        if component is not None:
            self.component = component
        super(AbstractOverlay, self).__init__(*args, **kw)

    def do_layout(self, size=None, force=False, component=None):
        """ Tells this component to do a layout at a given size.  This differs
        from the superclass Component.do_layout() in that it accepts an
        optional **component** argument.
        """
        if self.layout_needed or force:
            if size is not None:
                self.bounds = size
            self._do_layout(component)
            self._layout_needed = False
        for underlay in self.underlays:
            if underlay.visible or underlay.invisible_layout:
                underlay.do_layout(component)
        for overlay in self.overlays:
            if overlay.visible or overlay.invisible_layout:
                overlay.do_layout(component)

    def _draw(self, gc, view_bounds=None, mode="normal"):
        """ Draws the component, paying attention to **draw_order**.

        Overrides Component.
        """
        if self.component is not None:
            self.overlay(self.component, gc, view_bounds, mode)

    def _request_redraw(self):
        """ Overrides Enable Component.
        """
        if self.component is not None:
            self.component.request_redraw()
        super(AbstractOverlay, self)._request_redraw()
