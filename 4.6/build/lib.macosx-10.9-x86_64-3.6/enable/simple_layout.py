# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""Helper functions for a simple layout algorithm -- the same one used by
   OverlayPlotContainer.  Designed to be called from a container, but
   separated out because they are useful from ViewPort and Container"""


def simple_container_get_preferred_size(container, components=None):
    """ Returns the size (width,height) that is preferred for this component.

    Overrides PlotComponent.
    """
    if container.fixed_preferred_size is not None:
        return container.fixed_preferred_size

    if container.resizable == "":
        return container.outer_bounds

    if components is None:
        components = container.components

    fit_components = getattr(container, "fit_components", "")

    # this is used to determine if we should use our default bounds
    no_visible_components = True

    max_width = 0.0
    max_height = 0.0
    if fit_components != "":
        for component in components:
            if not container._should_layout(component):
                continue
            no_visible_components = False
            pref_size = None

            if "h" in fit_components:
                pref_size = component.get_preferred_size()
                if pref_size[0] > max_width:
                    max_width = pref_size[0]

            if "v" in fit_components:
                if pref_size is None:
                    pref_size = component.get_preferred_size()
                if pref_size[1] > max_height:
                    max_height = pref_size[1]

    if "h" not in fit_components:
        max_width = container.width
    if no_visible_components or (max_width == 0):
        max_width = container.default_size[0]

    if "v" not in fit_components:
        max_height = container.height
    if no_visible_components or (max_height == 0):
        max_height = container.default_size[1]

    # Add in our padding and border
    container._cached_preferred_size = (
        max_width + container.hpadding,
        max_height + container.vpadding,
    )
    return container._cached_preferred_size


def simple_container_do_layout(container, components=None):
    """ Actually performs a layout (called by do_layout()).
    """

    if components is None:
        components = container.components

    x = container.x
    y = container.y
    width = container.width
    height = container.height

    for component in components:
        if hasattr(container, "_should_layout"):
            if not container._should_layout(component):
                continue

        position = list(component.outer_position)
        bounds = list(component.outer_bounds)
        if "h" in component.resizable:
            position[0] = 0
            bounds[0] = width
        if "v" in component.resizable:
            position[1] = 0
            bounds[1] = height

        # Set both bounds at once.  This is a slight perforance fix because
        # it only fires two trait events instead of four.  It is also needed
        # in order for the event-based aspect ratio enforcement code to work.
        component.outer_position = position
        component.outer_bounds = bounds

    # Tell all of our components to do a layout
    for component in components:
        component.do_layout()
