# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from collections import deque

# traits imports
from traits.api import (
    Any,
    Bool,
    Callable,
    Dict,
    Either,
    Instance,
    List,
    Property,
)

# local imports
from .container import Container
from .coordinate_box import CoordinateBox
from .layout.layout_helpers import expand_constraints
from .layout.layout_manager import LayoutManager
from .layout.utils import (
    add_symbolic_contents_constraints,
    get_from_constraints_namespace,
)


class ConstraintsContainer(Container):
    """ A Container which lays out its child components using a
    constraints-based layout solver.

    """

    # A read-only symbolic object that represents the left boundary of
    # the component
    contents_left = Property(fget=get_from_constraints_namespace)

    # A read-only symbolic object that represents the right boundary
    # of the component
    contents_right = Property(fget=get_from_constraints_namespace)

    # A read-only symbolic object that represents the bottom boundary
    # of the component
    contents_bottom = Property(fget=get_from_constraints_namespace)

    # A read-only symbolic object that represents the top boundary of
    # the component
    contents_top = Property(fget=get_from_constraints_namespace)

    # A read-only symbolic object that represents the width of the
    # component
    contents_width = Property(fget=get_from_constraints_namespace)

    # A read-only symbolic object that represents the height of the
    # component
    contents_height = Property(fget=get_from_constraints_namespace)

    # A read-only symbolic object that represents the vertical center
    # of the component
    contents_v_center = Property(fget=get_from_constraints_namespace)

    # A read-only symbolic object that represents the horizontal
    # center of the component
    contents_h_center = Property(fget=get_from_constraints_namespace)

    # The layout constraints for this container.
    # This can either be a list or a callable. If it is a callable, it will be
    # called with a single argument, the ConstraintsContainer, and be expected
    # to return a list of constraints.
    layout_constraints = Either(List, Callable)

    # A boolean which indicates whether or not to allow the layout
    # ownership of this container to be transferred to an ancestor.
    # This is False by default, which means that every container
    # get its own layout solver. This improves speed and reduces
    # memory use (by keeping a solver's internal tableaux small)
    # but at the cost of not being able to share constraints
    # across Container boundaries. This flag must be explicitly
    # marked as True to enable sharing.
    share_layout = Bool(False)

    # Sharing related private traits
    _owns_layout = Bool(True)
    _layout_owner = Any

    # The contents box constraints for this container
    _contents_constraints = Property

    # The user-specified layout constraints, with layout helpers expanded
    _layout_constraints = Property

    # A dictionary of components added to this container
    _component_map = Dict

    # The kiwi solver
    _layout_manager = Instance(LayoutManager, allow_none=True)
    _offset_table = List
    _layout_table = List

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    def do_layout(self, size=None, force=False):
        """ Make sure child components get a chance to refresh their layout.
        """
        for component in self.components:
            component.do_layout(size=size, force=force)

    def refresh(self):
        """ Re-run the constraints solver in response to a resize or
        constraints modification.
        """
        if self._owns_layout:
            if self._layout_manager is None:
                return

            mgr_layout = self._layout_manager.layout
            offset_table = self._offset_table
            width_var = self.layout_width
            height_var = self.layout_height
            width, height = self.bounds

            def layout():
                running_index = 1
                for offset_index, item in self._layout_table:
                    dx, dy = offset_table[offset_index]
                    nx, ny = item.left.value(), item.bottom.value()
                    item.position = (nx - dx, ny - dy)
                    item.bounds = (
                        item.layout_width.value(),
                        item.layout_height.value(),
                    )
                    offset_table[running_index] = (nx, ny)
                    running_index += 1

            mgr_layout(layout, width_var, height_var, (width, height))

            self.invalidate_draw()
        else:
            self._layout_owner.refresh()

    def relayout(self):
        """ Explicitly regenerate the container's constraints and refresh the
        layout.
        """
        if not self.share_layout:
            self._init_layout()
            self.refresh()
        elif self._layout_owner is not None:
            self._layout_owner.relayout()

    # ------------------------------------------------------------------------
    # Layout Sharing
    # ------------------------------------------------------------------------
    def transfer_layout_ownership(self, owner):
        """ A method which can be called by other components in the
        hierarchy to gain ownership responsibility for the layout
        of the children of this container. By default, the transfer
        is allowed and is the mechanism which allows constraints to
        cross widget boundaries. Subclasses should reimplement this
        method if different behavior is desired.

        Parameters
        ----------
        owner : ConstraintsContainer
            The container which has taken ownership responsibility
            for laying out the children of this component. All
            relayout and refresh requests will be forwarded to this
            component.

        Returns
        -------
        results : bool
            True if the transfer was allowed, False otherwise.

        """
        if not self.share_layout:
            return False
        self._owns_layout = False
        self._layout_owner = owner
        self._layout_manager = None
        return True

    def will_transfer(self):
        """ Whether or not the container expects to transfer its layout
        ownership to its parent.

        This method is predictive in nature and exists so that layout
        managers are not senslessly created during the bottom-up layout
        initialization pass. It is declared public so that subclasses
        can override the behavior if necessary.

        """
        cls = ConstraintsContainer
        if self.share_layout:
            if self.container and isinstance(self.container, cls):
                return True
        return False

    # ------------------------------------------------------------------------
    # Traits methods
    # ------------------------------------------------------------------------
    def _bounds_changed(self, old, new):
        """ Run the solver when the container's bounds change.
        """
        super(ConstraintsContainer, self)._bounds_changed(old, new)
        self.refresh()

    def _layout_constraints_changed(self):
        """ Refresh the layout when the user constraints change.
        """
        self.relayout()

    def _get__contents_constraints(self):
        """ Return the constraints which define the content box of this
        container.

        """
        add_symbolic_contents_constraints(self._constraints_vars)

        return [
            self.contents_left == self.left,
            self.contents_bottom == self.bottom,
            self.contents_right == self.left + self.layout_width,
            self.contents_top == self.bottom + self.layout_height,
        ]

    def _get__layout_constraints(self):
        """ React to changes of the user controlled constraints.
        """
        if self.layout_constraints is None:
            return []

        if callable(self.layout_constraints):
            new = self.layout_constraints(self)
        else:
            new = self.layout_constraints

        # Expand any layout helpers
        return [cns for cns in expand_constraints(self, new)]

    def __components_items_changed(self, event):
        """ Make sure components that are added can be used with constraints.
        """
        # Remove stale components from the map
        for item in event.removed:
            item.on_trait_change(
                self._component_size_hint_changed,
                "layout_size_hint",
                remove=True,
            )
            del self._component_map[item.id]

        # Check the added components
        self._check_and_add_components(event.added)

    def __components_changed(self, new):
        """ Make sure components that are added can be used with constraints.
        """
        # Clear the component maps
        for key, item in self._component_map.items():
            item.on_trait_change(
                self._component_size_hint_changed,
                "layout_size_hint",
                remove=True,
            )
        self._component_map = {}

        # Check the new components
        self._check_and_add_components(new)

    def _component_size_hint_changed(self):
        """ Refresh the size hint contraints for a child component
        """
        self.relayout()

    # ------------------------------------------------------------------------
    # Protected methods
    # ------------------------------------------------------------------------

    def _build_layout_table(self):
        """ Build the layout and offset tables for this container.

        A layout table is a pair of flat lists which hold the required
        objects for laying out the child widgets of this container.
        The flat table is built in advance (and rebuilt if and when
        the tree structure changes) so that it's not necessary to
        perform an expensive tree traversal to layout the children
        on every resize event.

        Returns
        -------
        result : (list, list)
            The offset table and layout table to use during a resize
            event.

        """
        # The offset table is a list of (dx, dy) tuples which are the
        # x, y offsets of children expressed in the coordinates of the
        # layout owner container. This owner container may be different
        # from the parent of the widget, and so the delta offset must
        # be subtracted from the computed geometry values during layout.
        # The offset table is updated during a layout pass in breadth
        # first order.
        #
        # The layout table is a flat list of (idx, updater) tuples. The
        # idx is an index into the offset table where the given child
        # can find the offset to use for its layout. The updater is a
        # callable provided by the widget which accepts the dx, dy
        # offset and will update the layout geometry of the widget.
        zero_offset = (0, 0)
        offset_table = [zero_offset]
        layout_table = []
        queue = deque((0, child) for child in self._component_map.values())

        # Micro-optimization: pre-fetch bound methods and store globals
        # as locals. This method is not on the code path of a resize
        # event, but it is on the code path of a relayout. If there
        # are many children, the queue could potentially grow large.
        push_offset = offset_table.append
        push_item = layout_table.append
        push = queue.append
        pop = queue.popleft
        CoordinateBox_ = CoordinateBox
        Container_ = ConstraintsContainer
        isinst = isinstance

        # The queue yields the items in the tree in breadth-first order
        # starting with the immediate children of this container. If a
        # given child is a container that will share its layout, then
        # the children of that container are added to the queue to be
        # added to the layout table.
        running_index = 0
        while queue:
            offset_index, item = pop()
            if isinst(item, CoordinateBox_):
                push_item((offset_index, item))
                push_offset(zero_offset)
                running_index += 1
                if isinst(item, Container_):
                    if item.transfer_layout_ownership(self):
                        for child in item._component_map.values():
                            push((running_index, child))

        return offset_table, layout_table

    def _check_and_add_components(self, components):
        """ Make sure components can be used with constraints.
        """
        for item in components:
            key = item.id
            if len(key) == 0:
                msg = "Components added to a {0} must have a valid 'id' trait."
                name = type(self).__name__
                raise ValueError(msg.format(name))
            elif key in self._component_map:
                msg = "A Component with id '{0}' has already been added."
                raise ValueError(msg.format(key))
            elif key == self.id:
                msg = "Can't add a Component with the same id as its parent."
                raise ValueError(msg)

            self._component_map[key] = item
            item.on_trait_change(
                self._component_size_hint_changed, "layout_size_hint"
            )

        # Update the layout
        self.relayout()

    def _generate_constraints(self, layout_table):
        """ Creates the list of kiwi Constraint objects for
        the widgets for which this container owns the layout.

        This method walks over the items in the given layout table and
        aggregates their constraints into a single list of kiwi
        Constraint objects which can be given to the layout
        manager.

        Parameters
        ----------
        layout_table : list
            The layout table created by a call to _build_layout_table.

        Returns
        -------
        result : list
            The list of kiwi Constraint instances to pass to
            the layout manager.

        """
        user_cns = self._layout_constraints
        user_cns_extend = user_cns.extend

        # The list of raw kiwi constraints which will be returned
        # from this method to be added to the kiwi solver.
        raw_cns = self._hard_constraints + self._contents_constraints
        raw_cns_extend = raw_cns.extend

        isinst = isinstance
        Container_ = ConstraintsContainer
        # The first element in a layout table item is its offset index
        # which is not relevant to constraints generation.
        for _, child in layout_table:
            raw_cns_extend(child._hard_constraints)
            if isinst(child, Container_):
                if child.transfer_layout_ownership(self):
                    user_cns_extend(child._layout_constraints)
                    raw_cns_extend(child._contents_constraints)
                else:
                    raw_cns_extend(child._size_constraints)
            else:
                raw_cns_extend(child._size_constraints)

        return raw_cns + user_cns

    def _init_layout(self):
        """ Initializes the layout for the container.

        """
        # Layout ownership can only be transferred *after* this init
        # layout method is called, since layout occurs bottom up. So,
        # we only initialize a layout manager if we are not going to
        # transfer ownership at some point.
        if not self.will_transfer():
            offset_table, layout_table = self._build_layout_table()
            cns = self._generate_constraints(layout_table)
            # Initializing the layout manager can fail if the objective
            # function is unbounded. We let that failure occur so it can
            # be logged. Nothing is stored until it succeeds.
            manager = LayoutManager()
            manager.initialize(cns)
            self._offset_table = offset_table
            self._layout_table = layout_table
            self._layout_manager = manager
