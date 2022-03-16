.. _writing_a_backend:

Writing A Kiva Backend
======================

To write a Kiva backend you need to implement all the abstract methods of the
:py:class:`.AbstractGraphicsContext`.  Once this is done, instances of the new
backend should be able to be used in most places that one of the current
backends is.  Most sophisticated 2D drawing libraries are very similar to
Kiva in terms of the facilities they offer, and so in many cases this is just
a matter of translating between the Kiva API and the underlying library.

In some cases, however, the underlying drawing capabilities are more primitive.

GraphicsContextBase
-------------------

Kiva provides a core set of functionality in
:py:class:`~kiva.basecore2d.GraphicsContextBase` that makes it easier to
implement the full Kiva API on top of a more basic drawing API (for example
one which can't draw Bezier curves or arcs, or have a notion of graphics
state).  To use this class, you need to subclass and implement a number of
``device_`` methods that actually perform the drawing operations.  These
include:

- ``device_update_fill_state``
- ``device_update_line_state``
- ``device_fill_points``
- ``device_stroke_points``
- ``device_draw_image``
- ``device_show_text``
- ``device_get_full_text_extent``
- ``device_set_clipping_path``
- ``device_destroy_clipping_path``

The :py:class:`~kiva.basecore2d.GraphicsContextBase` class tracks all the
required state internally, including saving and restoring state.  However the
device code needs to be able to handle applying the current affine
transformations and styling when drawing polygons, text and images, either in
the implementation code or in the underlying graphics library.
