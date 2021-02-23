# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# Enthought library imports
from traits.api import HasTraits


class AbstractRenderController(HasTraits):
    def draw(self, component, gc, view_bounds=None, mode="normal"):
        raise NotImplementedError


class RenderController(AbstractRenderController):
    """ The default Enable render controller for components """

    # The default list of available layers.
    LAYERS = [
        "background",
        "image",
        "underlay",
        "component",
        "overlay",
        "border",
    ]

    def draw(self, component, gc, view_bounds=None, mode="normal"):
        if component.visible:
            for layer in self.LAYERS:
                func = getattr(component, "_draw_" + layer, None)
                if func:
                    func(gc, view_bounds, mode)


class OldEnableRenderController(AbstractRenderController):
    """
    Performs rendering of components and containers in the default way
    that Enable used to, prior to the encapsulation of rendering in
    RenderControllers.

    Note that containers have a default implementation of _draw() that
    in turn calls _draw_container(), which is pass-through in the base class.
    """

    def draw(self, component, gc, view_bounds=None, mode="normal"):
        component._draw_background(gc, view_bounds, mode)
        component._draw(gc, view_bounds, mode)
        component._draw_border(gc, view_bounds, mode)


class OldChacoRenderController(AbstractRenderController):
    """ Performs render the way that it was done before the draw_order
    changes were implemented.

    This has the name Chaco in it because this functionality used to be
    in Chaco; however, with configurable rendering now in Enable, this
    class resides in the Enable package, and will eventually be deprecated.
    """

    def draw(self, component, gc, view_bounds=None, mode="normal"):
        if component.visible:
            # Determine if the component has an active tool and if
            # we need to transfer execution to it
            tool = component._active_tool
            if tool is not None and tool.draw_mode == "overlay":
                tool.draw(gc, view_bounds)
            else:
                if component.use_backbuffer:
                    if mode == "overlay":
                        # Since kiva currently doesn't support blend modes,
                        # if we have to draw in overlay mode, we have to draw
                        # normally.
                        self._do_draw(component, gc, view_bounds, mode)
                        component._backbuffer = None
                        component.invalidate_draw()
                    if not self.draw_valid:
                        from .kiva_graphics_context import GraphicsContext

                        bb = GraphicsContext(tuple(map(int, component.bounds)))
                        bb.translate_ctm(-component.x, -component.y)
                        self._do_draw(
                            component, bb, view_bounds=None, mode=mode
                        )
                        component._backbufer = bb
                        component.draw_valid = True

                    gc.draw_imge(
                        component._backbuffer,
                        component.position + component.bounds,
                    )

                else:
                    self._do_draw(component, gc, view_bounds, mode)

    def _do_draw(self, component, gc, view_bounds=None, mode="normal"):
        pass
