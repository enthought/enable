# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Containers which lay out their components horizontally or vertically

"""

from traits.api import Enum, Float

from .container import Container
from .stacked_layout import stacked_preferred_size, stack_layout


class StackedContainer(Container):
    """ Base class for stacked containers
    """

    # The dimension along which to stack components that are added to
    # this container.
    stack_dimension = Enum("h", "v")

    # The "other" dimension, i.e., the dual of the stack dimension.
    other_dimension = Enum("v", "h")

    # The index into obj.position and obj.bounds that corresponds to
    # **stack_dimension**.  This is a class-level and not an instance-level
    # attribute. It must be 0 or 1.
    stack_index = 0

    # The amount of space to put between components.
    spacing = Float(0.0)

    def get_preferred_size(self, components=None):
        return stacked_preferred_size(self, components)


class HStackedContainer(StackedContainer):
    """
    A container that stacks components horizontally.
    """

    # Overrides StackedPlotContainer.
    stack_dimension = "h"
    # Overrides StackedPlotContainer.
    other_dimension = "v"
    # Overrides StackedPlotContainer.
    stack_index = 0

    # VPlotContainer attributes

    # The horizontal alignment of objects that don't span the full width.
    halign = Enum("bottom", "top", "center")

    # The order in which components in the plot container are laid out.
    stack_order = Enum("left_to_right", "right_to_left")

    def _do_layout(self):
        """ Actually performs a layout (called by do_layout()).
        """
        if self.stack_order == "left_to_right":
            components = self.components
        else:
            components = self.components[::-1]
        if self.halign == "bottom":
            align = "min"
        elif self.halign == "center":
            align = "center"
        else:
            align = "max"

        return stack_layout(self, components, align)


class VStackedContainer(StackedContainer):
    """
    A container that stacks components vertically.
    """

    # Overrides StackedPlotContainer.
    stack_dimension = "v"
    # Overrides StackedPlotContainer.
    other_dimension = "h"
    # Overrides StackedPlotContainer.
    stack_index = 1

    # VPlotContainer attributes

    # The horizontal alignment of objects that don't span the full width.
    halign = Enum("left", "right", "center")

    # The order in which components in the plot container are laid out.
    stack_order = Enum("bottom_to_top", "top_to_bottom")

    def _do_layout(self):
        """ Actually performs a layout (called by do_layout()).
        """
        if self.stack_order == "bottom_to_top":
            components = self.components
        else:
            components = self.components[::-1]
        if self.halign == "left":
            align = "min"
        elif self.halign == "center":
            align = "center"
        else:
            align = "max"

        return stack_layout(self, components, align)
