# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Routines for stacked layout of components in a container
"""

# TODO: stolen from Chaco PlotContainers, should change their classes to use


def stacked_preferred_size(container, components=None):
    """ Returns the size (width,height) that is preferred for this component.

    Overrides Component.
    """
    if container.fixed_preferred_size is not None:
        container._cached_preferred_size = container.fixed_preferred_size
        return container.fixed_preferred_size

    if components is None:
        components = container.components

    ndx = container.stack_index
    other_ndx = 1 - ndx

    no_visible_components = True
    total_size = 0
    max_other_size = 0
    for component in components:
        if not container._should_layout(component):
            continue

        no_visible_components = False

        pref_size = component.get_preferred_size()
        total_size += pref_size[ndx] + container.spacing
        if pref_size[other_ndx] > max_other_size:
            max_other_size = pref_size[other_ndx]

    if total_size >= container.spacing:
        total_size -= container.spacing

    if (container.stack_dimension not in container.resizable
            and container.stack_dimension not in container.fit_components):
        total_size = container.bounds[ndx]
    elif no_visible_components or (total_size == 0):
        total_size = container.default_size[ndx]

    if (container.other_dimension not in container.resizable
            and container.other_dimension not in container.fit_components):
        max_other_size = container.bounds[other_ndx]
    elif no_visible_components or (max_other_size == 0):
        max_other_size = container.default_size[other_ndx]

    if ndx == 0:
        container._cached_preferred_size = (
            total_size + container.hpadding,
            max_other_size + container.vpadding,
        )
    else:
        container._cached_preferred_size = (
            max_other_size + container.hpadding,
            total_size + container.vpadding,
        )

    return container._cached_preferred_size


def stack_layout(container, components, align):
    """ Helper method that does the actual work of layout.
    """

    size = list(container.bounds)
    if container.fit_components != "":
        container.get_preferred_size()
        if "h" in container.fit_components:
            size[0] = container._cached_preferred_size[0] - container.hpadding
        if "v" in container.fit_components:
            size[1] = container._cached_preferred_size[1] - container.vpadding

    ndx = container.stack_index
    other_ndx = 1 - ndx
    other_dim = container.other_dimension

    # Assign sizes of non-resizable components, and compute the total size
    # used by them (along the stack dimension).
    total_fixed_size = 0
    resizable_components = []
    size_prefs = {}
    total_resizable_size = 0

    for component in components:
        if not container._should_layout(component):
            continue
        if container.stack_dimension not in component.resizable:
            total_fixed_size += component.outer_bounds[ndx]
        else:
            preferred_size = component.get_preferred_size()
            size_prefs[component] = preferred_size
            total_resizable_size += preferred_size[ndx]
            resizable_components.append(component)

    new_bounds_dict = {}

    # Assign sizes of all the resizable components along the stack dimension
    if resizable_components:
        space = container.spacing * (len(container.components) - 1)
        avail_size = size[ndx] - total_fixed_size - space
        if total_resizable_size > 0:
            scale = avail_size / float(total_resizable_size)
            for component in resizable_components:
                tmp = list(component.outer_bounds)
                tmp[ndx] = int(size_prefs[component][ndx] * scale)
                new_bounds_dict[component] = tmp
        else:
            each_size = int(avail_size / len(resizable_components))
            for component in resizable_components:
                tmp = list(component.outer_bounds)
                tmp[ndx] = each_size
                new_bounds_dict[component] = tmp

    # Loop over all the components, assigning position and computing the
    # size in the other dimension and its position.
    cur_pos = 0
    for component in components:
        if not container._should_layout(component):
            continue

        position = list(component.outer_position)
        position[ndx] = cur_pos

        bounds = new_bounds_dict.get(component, list(component.outer_bounds))
        cur_pos += bounds[ndx] + container.spacing

        if (bounds[other_ndx] > size[other_ndx]
                or other_dim in component.resizable):
            # If the component is resizable in the other dimension or it
            # exceeds the container bounds, set it to the maximum size of the
            # container

            position[other_ndx] = 0
            bounds[other_ndx] = size[other_ndx]
        else:
            position[other_ndx] = 0
            if align == "min":
                pass
            elif align == "max":
                position[other_ndx] = size[other_ndx] - bounds[other_ndx]
            elif align == "center":
                position[other_ndx] = (
                    size[other_ndx] - bounds[other_ndx]
                ) / 2.0

        component.outer_position = position
        component.outer_bounds = bounds
        component.do_layout()
