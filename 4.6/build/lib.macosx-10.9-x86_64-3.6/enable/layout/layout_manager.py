# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from contextlib import contextmanager

import kiwisolver as kiwi


class LayoutManager(object):
    """ A class which uses a kiwi solver to manage a system
    of constraints.
    """

    def __init__(self):
        self._solver = kiwi.Solver()
        self._edit_stack = []
        self._initialized = False
        self._running = False

    def initialize(self, constraints):
        """ Initialize the solver with the given constraints.

        Parameters
        ----------
        constraints : Iterable
            An iterable that yields the constraints to add to the
            solvers.
        """
        if self._initialized:
            raise RuntimeError("Solver already initialized")
        solver = self._solver
        for cn in constraints:
            solver.addConstraint(cn)
        self._initialized = True

    def replace_constraints(self, old_cns, new_cns):
        """ Replace constraints in the solver.

        Parameters
        ----------
        old_cns : list
            The list of kiwi constraints to remove from the
            solver.

        new_cns : list
            The list of kiwi constraints to add to the solver.
        """
        if not self._initialized:
            raise RuntimeError("Solver not yet initialized")
        solver = self._solver
        for cn in old_cns:
            solver.removeConstraint(cn)
        for cn in new_cns:
            solver.addConstraint(cn)

    def layout(self, cb, width, height, size, strength=kiwi.strength.medium):
        """ Perform an iteration of the solver for the new width and
        height constraint variables.

        Parameters
        ----------
        cb : callable
            A callback which will be called when new values from the
            solver are available. This will be called from within a
            solver context while the solved values are valid. Thus
            the new values should be consumed before the callback
            returns.

        width : Constraint Variable
            The constraint variable representing the width of the
            main layout container.

        height : Constraint Variable
            The constraint variable representing the height of the
            main layout container.

        size : (int, int)
            The (width, height) size tuple which is the current size
            of the main layout container.

        strength : kiwisolver strength, optional
            The strength with which to perform the layout using the
            current size of the container. i.e. the strength of the
            resize. The default is kiwisolver.strength.medium.
        """
        if not self._initialized:
            raise RuntimeError("Layout with uninitialized solver")
        if self._running:
            return
        try:
            self._running = True
            w, h = size
            solver = self._solver
            pairs = ((width, strength), (height, strength))
            with self._edit_context(pairs):
                solver.suggestValue(width, w)
                solver.suggestValue(height, h)
                solver.updateVariables()
                cb()
        finally:
            self._running = False

    def get_min_size(self, width, height, strength=kiwi.strength.medium):
        """ Run an iteration of the solver with the suggested size of the
        component set to (0, 0). This will cause the solver to effectively
        compute the minimum size that the window can be to solve the
        system.

        Parameters
        ----------
        width : Constraint Variable
            The constraint variable representing the width of the
            main layout container.

        height : Constraint Variable
            The constraint variable representing the height of the
            main layout container.

        strength : kiwisolver strength, optional
            The strength with which to perform the layout using the
            current size of the container. i.e. the strength of the
            resize. The default is kiwisolver.strength.medium.

        Returns
        -------
        result : (float, float)
            The floating point (min_width, min_height) size of the
            container which would best satisfy the set of constraints.
        """
        if not self._initialized:
            raise RuntimeError("Get min size on uninitialized solver")
        solver = self._solver
        pairs = ((width, strength), (height, strength))
        with self._edit_context(pairs):
            solver.suggestValue(width, 0.0)
            solver.suggestValue(height, 0.0)
            solver.updateVariables()
            min_width = width.value()
            min_height = height.value()
        return (min_width, min_height)

    def get_max_size(self, width, height, strength=kiwi.strength.medium):
        """ Run an iteration of the solver with the suggested size of
        the component set to a very large value. This will cause the
        solver to effectively compute the maximum size that the window
        can be to solve the system. The return value is a tuple numbers.
        If one of the numbers is -1, it indicates there is no maximum in
        that direction.

        Parameters
        ----------
        width : Constraint Variable
            The constraint variable representing the width of the
            main layout container.

        height : Constraint Variable
            The constraint variable representing the height of the
            main layout container.

        strength : kiwisolver strength, optional
            The strength with which to perform the layout using the
            current size of the container. i.e. the strength of the
            resize. The default is kiwisolver.strength.medium.

        Returns
        -------
        result : (float or -1, float or -1)
            The floating point (max_width, max_height) size of the
            container which would best satisfy the set of constraints.
        """
        if not self._initialized:
            raise RuntimeError("Get max size on uninitialized solver")
        max_val = 2 ** 24 - 1  # Arbitrary, but the max allowed by Qt.
        solver = self._solver
        pairs = ((width, strength), (height, strength))
        with self._edit_context(pairs):
            solver.suggestValue(width, max_val)
            solver.suggestValue(height, max_val)
            solver.updateVariables()
            max_width = width.value()
            max_height = width.value()
        width_diff = abs(max_val - int(round(max_width)))
        height_diff = abs(max_val - int(round(max_height)))
        if width_diff <= 1:
            max_width = -1
        if height_diff <= 1:
            max_height = -1
        return (max_width, max_height)

    def _push_edit_vars(self, pairs):
        """ Push edit variables into the solver.

        The current edit variables will be removed and the new edit
        variables will be added.

        Parameters
        ----------
        pairs : sequence
            A sequence of 2-tuples of (var, strength) which should be
            added as edit variables to the solver.
        """
        solver = self._solver
        stack = self._edit_stack
        if stack:
            for v, strength in stack[-1]:
                solver.removeEditVariable(v)
        stack.append(pairs)
        for v, strength in pairs:
            solver.addEditVariable(v, strength)

    def _pop_edit_vars(self):
        """ Restore the previous edit variables in the solver.

        The current edit variables will be removed and the previous
        edit variables will be re-added.
        """
        solver = self._solver
        stack = self._edit_stack
        for v, strength in stack.pop():
            solver.removeEditVariable(v)
        if stack:
            for v, strength in stack[-1]:
                solver.addEditVariable(v, strength)

    @contextmanager
    def _edit_context(self, pairs):
        """ A context manager for temporary solver edits.

        This manager will push the edit vars into the solver and pop
        them when the context exits.

        Parameters
        ----------
        pairs : list
            A list of 2-tuple of (var, strength) which should be added
            as temporary edit variables to the solver.
        """
        self._push_edit_vars(pairs)
        yield
        self._pop_edit_vars()
