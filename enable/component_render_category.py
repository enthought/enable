"""
This module contains the RenderingCategory, which defines the rendering aspect of
components.  It uses Traits Categories to extend the Component class defined
in component.py.

NOTE: This means that you should import enable.Component from enable.api, and
not directly from component.py.
"""

from __future__ import with_statement

# Enthought library imports
from traits.api import Any, Bool, Category, Enum, Instance, \
                                 Int, List

# Local relative imports
from colors import black_color_trait, white_color_trait
from abstract_component import AbstractComponent
from enable_traits import LineStyle
from render_controllers import AbstractRenderController, OldEnableRenderController, \
                               RenderController


# Singleton representing the default Enable render controller
DefaultRenderController = OldEnableRenderController()


class ComponentRenderCategory(Category, AbstractComponent):
    """ Encapsulates the rendering-related aspects of a component and
    extends the Component class in component.py.

    !! This class should not be instantiated or subclassed !!  Please refer
    to traits.Category for more information.
    """

    # The controller that determines how this component renders.  By default,
    # this is the singleton
    render_controller = Instance(AbstractRenderController, factory = DefaultRenderController)

    # Is the component visible?
    visible = Bool(True)

    # Does this container prefer to draw all of its components in one pass, or
    # does it prefer to cooperate in its container's layer-by-layer drawing?
    # If unified_draw is on, then this component will draw as a unified whole,
    # and its parent container will call our _draw() method when drawing the
    # layer indicated in self.draw_layer.
    # If unified_draw is off, then our parent container will call
    # self._dispatch_draw() with the name of each layer as it goes through its
    # list of layers.
    unified_draw = Bool(False)

    # A list of the order in which various layers of this component
    # should be rendered.  This is only used if the component does
    # unified draw.
    draw_order = List

    # If unified_draw is True for this component, then this attribute
    # determines what layer it will be drawn on.  This is used by containers
    # and external classes whose drawing loops will call this component.
    draw_layer = Enum(RenderController.LAYERS)

    #------------------------------------------------------------------------
    # Background and padding
    #------------------------------------------------------------------------

    # Should the padding area be filled with the background color?
    fill_padding = Bool(False)

    # The background color of this component.  By default all components have
    # a white background.  This can be set to "transparent" or "none" if the
    # component should be see-through.
    bgcolor = white_color_trait

    #------------------------------------------------------------------------
    # Border traits
    #------------------------------------------------------------------------

    # The width of the border around this component.  This is taken into account
    # during layout, but only if the border is visible.
    border_width = Int(1)

    # Is the border visible?  If this is false, then all the other border
    # properties are not
    border_visible = Bool(False)

    # The line style (i.e. dash pattern) of the border.
    border_dash = LineStyle

    # The color of the border.  Only used if border_visible is True.
    border_color = black_color_trait

    # Should the border be drawn as part of the overlay or the background?
    overlay_border = Bool(True)

    # Should the border be drawn inset (on the plot) or outside the plot
    # area?
    inset_border = Bool(True)


    #------------------------------------------------------------------------
    # Backbuffer traits
    #------------------------------------------------------------------------

    # Should this component do a backbuffered draw, i.e. render itself
    # to an offscreen buffer that is cached for later use?  If False,
    # then the component will never render itself backbufferd, even
    # if explicitly asked to do so.
    use_backbuffer = Bool(False)

    # Reflects the validity state of the backbuffer.  This is usually set by
    # the component itself or set on the component by calling
    # _invalidate_draw().  It is exposed as a public trait for the rare cases
    # when another components wants to know the validity of this component's
    # backbuffer, i.e. if a draw were to occur, whether the component would
    # actually change.
    draw_valid = Bool(False)

    # Should the backbuffer include the padding area?
    # TODO: verify that backbuffer invalidation occurs if this attribute
    # is changed.
    backbuffer_padding = Bool(True)

    #------------------------------------------------------------------------
    # Private traits
    #------------------------------------------------------------------------

    _backbuffer = Any


    def draw(self, gc, view_bounds=None, mode="default"):
        """
        Renders this component onto a GraphicsContext.

        "view_bounds" is a 4-tuple (x, y, dx, dy) of the viewed region relative
        to the CTM of the gc.

        "mode" can be used to require this component render itself in a particular
        fashion, and can be "default" or any of the enumeration values of
        self.default_draw_mode.
        """
        self.render_controller.draw(self, gc, view_bounds, mode)
        return

    def _draw_component(self, gc, view_bounds=None, mode="default"):
        """ This function actually draws the core parts of the component
        itself, i.e. the parts that belong on the "main" layer.  Subclasses
        should implement this.
        """
        pass

    def _draw_border(self, gc, view_bounds=None, mode="default"):
        """ Utility method to draw the borders around this component """
        if not self.border_visible:
            return

        border_width = self.border_width
        with gc:
            gc.set_line_width(border_width)
            gc.set_line_dash(self.border_dash_)
            gc.set_stroke_color(self.border_color_)
            gc.begin_path()
            gc.rect(self.x - border_width/2.0,
                    self.y - border_width/2.0,
                    self.width + 2*border_width - 1,
                    self.height + 2*border_width - 1)
            gc.stroke_path()
        return

    def _draw_background(self, gc, view_bounds=None, mode="default"):
        if self.bgcolor not in ("transparent", "none"):
            gc.set_fill_color(self.bgcolor_)
            gc.rect(*(self.position + self.bounds))
            gc.fill_path()
        return
