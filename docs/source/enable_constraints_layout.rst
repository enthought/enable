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

Layout Helpers
--------------

In practice, it's too tedious to specify all the constraints for a rich UI
layout. To aid in the generation of layouts, the layout helpers from Enaml_ are
also available in Enable. The layout helpers are:

 * :func:`horizontal`: Takes a list of components and lines them up using their left and right edges.
 * :func:`vertical`: Takes a list of components and lines them up using their top and bottom edges.
 * :func:`hbox`: Like :func:`horizontal`, but ensures the height of components matches the container.
 * :func:`vbox`: Like :func:`vertical`, but ensures the width of components matches the container.
 * :func:`align`: Aligns a single constraint across multiple components.
 * :func:`grid`: Creates an NxM grid of components. Components may span multiple columns or rows.
 * :func:`spacer`: Creates space between two adjacent components.


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
