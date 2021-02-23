.. _constraints-layout:

Enable Constraints Layout
=========================

This document describes the constraints-based layout system that is being
proposed as the new layout model going forward. Familiarity with Enaml_ and
its layout system is helpful but not required.


Using Constraints
-----------------

:class:`ConstraintsContainer` is a :class:`Container` subclass which uses the
Cassowary_ constraint solver to determine the layout of its child
:class:`Component` instances. This is achieved by adding constraint variables
to the :class:`Component` class which define a simple box model:

* :attr:`layout_height`: The height of the component.
* :attr:`layout_width`: The width of the component.
* :attr:`left`: The left edge of the component.
* :attr:`right`: The right edge of the component.
* :attr:`top`: The top edge of the component.
* :attr:`bottom`: The bottom edge of the component.
* :attr:`h_center`: The vertical center line between the left and right edges
* :attr:`v_center`: The  horizontal center line between the top and bottom edges

Additionally, there are some constraints which only exist on
:class:`ConstraintsContainer`:

* :attr:`contents_height`: The height of the container.
* :attr:`contents_width`: The width of the container.
* :attr:`contents_left`: The left edge of the container.
* :attr:`contents_right`: The right edge of the container.
* :attr:`contents_top`: The top edge of the container.
* :attr:`contents_bottom`: The bottom edge of the container.
* :attr:`contents_h_center`: The vertical center line of the container.
* :attr:`contents_v_center`: The  horizontal center line of the container.

These variables can be used in linear inequality expressions which make up the
layout constraints of a container:

::

  def build_hierarchy():
    container = ConstraintsContainer()
    one = Component()
    two = Component()
    container.add(one, two)
    container.layout_constraints = [
        one.layout_width == two.layout_width * 2.0,
        one.layout_height == two.layout_height,
        # ... and so on ...
    ]

    return container

For more complicated layouts, the :attr:`layout_constraints` trait on a
:class:`ConstraintsContainer` can be a :class:`callable`. The function is
passed a reference to the container and should return a list of
:class:`LinearContraints` objects or layout helper instances (as described below).

::

   def create_container(self):
    self.container = ConstraintsContainer()
    self.container.add(self.bar)
    self.container.layout_constraints = self.my_layout_constraints

  def my_layout_constraints(self, container):
    cns = []

    if self.foo:
      cns.append(self.foo.layout_height <= 300)
      cns.append(hbox(self.foo, self.bar))
    cns.append(self.bar.layout_width == 250)

    return cns

If :attr:`layout_constraints` is callable, it will be invoked each time a
component is added to the container or whenever the :attr:`layout_size_hint`
trait changes on a child component.

Layout Helpers
--------------

In practice, it's too tedious to specify all the constraints for a rich UI
layout. To aid in the generation of layouts, the layout helpers from Enaml_ are
also available in Enable. The layout helpers are:

:data:`spacer`: Creates space between two adjacent components.

.. function:: horizontal(*components[, spacing=10])

   Takes a list of components and lines them up using their left and right edges.

   :param components: A sequence of :class:`Component` or :class:`spacer` objects.
   :param spacing: How many pixels of inter-element spacing to use
   :type spacing: integer >= 0

.. function:: vertical(*components[, spacing=10])

   Takes a list of components and lines them up using their top and bottom edges.

   :param components: A sequence of :class:`Component` or :class:`spacer` objects.
   :param spacing: How many pixels of inter-element spacing to use
   :type spacing: integer >= 0

.. function:: hbox(*components[, spacing=10, margins=...])

   Like :func:`horizontal`, but ensures the height of components matches the container.

   :param components: A sequence of :class:`Component` or :class:`spacer` objects.
   :param spacing: How many pixels of inter-element spacing to use
   :type spacing: integer >= 0
   :param margins: An `int`, `tuple` of ints, or :class:`Box` of ints >= 0 which
                   indicate how many pixels of margin to add around the bounds
                   of the box. The default is 0.

.. function:: vbox(*components[, spacing=10, margins=...])

   Like :func:`vertical`, but ensures the width of components matches the container.

   :param components: A sequence of :class:`Component` or :class:`spacer` objects.
   :param spacing: How many pixels of inter-element spacing to use
   :type spacing: integer >= 0
   :param margins: An `int`, `tuple` of ints, or :class:`Box` of ints >= 0 which
                   indicate how many pixels of margin to add around the bounds
                   of the box. The default is 0.

.. function:: align(anchor, *components[, spacing=10])

   Aligns a single constraint across multiple components.

   :param anchor: The name of a constraint variable that exists on all of the
                  `components`.
   :param components: A sequence of :class:`Component` objects. Spacers are not allowed.
   :param spacing: How many pixels of inter-element spacing to use
   :type spacing: integer >= 0

.. function:: grid(*rows[, row_align='', row_spacing=10, column_align='', column_spacing=10, margins=...])

   Creates an NxM grid of components. Components may span multiple columns or rows.

   :param rows: A sequence of sequences of :class:`Component` objects
   :param row_align: The name of a constraint variable on an item. If given,
                     it is used to add constraints on the alignment of items
                     in a row. The constraints will only be applied to items
                     that do not span rows.
   :type row_align: string
   :param row_spacing: Indicates how many pixels of space should be placed
                       between rows in the grid. The default is 10.
   :type row_spacing: integer >= 0

   :param column_align: The name of a constraint variable on an item. If given,
                        it is used to add constraints on the alignment of items
                        in a column. The constraints will only be applied to
                        items that do not span columns.
   :type column_align: string
   :param column_spacing: Indicates how many pixels of space should be placed
                          between columns in the grid. The default is 10.
   :type column_spacing: integer >= 0
   :param margins: An `int`, `tuple` of ints, or :class:`Box` of ints >= 0 which
                   indicate how many pixels of margin to add around the bounds
                   of the box. The default is 0.


Fine Tuning Layouts
-------------------

:class:`Component` defines a :class:`Tuple` trait :attr:`layout_size_hint` which
controls the minimum size of a component when it's part of a contraints layout.
Additionally, :class:`Component` defines some strength traits that can be used
to fine tune the behavior of a component instance during layout. They are:

* :attr:`hug_height`: How strongly a component prefers the height of its size hint when it could grow.
* :attr:`hug_width`: How strongly a component prefers the width of its size hint when it could grow.
* :attr:`resist_height`: How strongly a component resists its height being made smaller than its size hint.
* :attr:`resist_width`: How strongly a component resists its width being made smaller than its size hint.

The allow values for these strengths are: `'required'`, `'strong'`, `'medium'`,
and `'weak'`.

.. _Cassowary: http://www.cs.washington.edu/research/constraints/cassowary/
.. _Enaml: http://docs.enthought.com/enaml/
