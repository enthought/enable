# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from traits.api import Enum
from enable.api import Component


class DrawingTool(Component):
    """
    A drawing tool is just a component that also defines a certain drawing mode
    so that its container knows how to render it and pass control to it.

    The DrawingTool base class also defines a draw() dispatch, so that
    different draw methods are called depending on the event state of the tool.
    """

    # The mode in which this tool draws:
    #
    # "normal"
    #     The tool draws like a normal component, alongside other components
    #     in the container
    # "overlay"
    #     The tool draws on top of over components in the container
    # "exclusive"
    #     The tool gets total control of how the container should be rendered
    draw_mode = Enum("normal", "overlay", "exclusive")

    def reset(self):
        """
        Causes the tool to reset any saved state and revert its event_state
        back to the initial value (usually "normal").
        """
        pass

    def complete_left_down(self, event):
        """
        Default function that causes the tool to reset if the user starts
        drawing again.
        """
        self.reset()
        self.request_redraw()
        self.normal_left_down(event)

    def _draw_mainlayer(self, gc, view_bounds, mode="default"):
        draw_func = getattr(self, self.event_state + "_draw", None)
        if draw_func:
            draw_func(gc)

    def request_redraw(self):
        if self.container is not None:
            self.container.request_redraw()
        elif hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.request_redraw()
        else:
            pass
