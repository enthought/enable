Kiva Interface Quick Reference
==============================

This document is a summary of the classes and functions available in
Kiva.  More specifically, it describes some of the details of the
kiva.agg backend, including enumerated types and helper classes.

Graphics Context
----------------

Construction
~~~~~~~~~~~~
__init__(ary_or_size, pix_format="bgra32", base_pixel_scale=1.0):
    ``ary_or_size`` can be either a numpy array or a tuple of the form
    (width, height). If it is an array, it will be used as the backing store
    for the pixels. **Its shape must be compatible with** ``pix_format``

    ``pix_format`` determines the pixel format and is a string which can be any
    of the following: "gray8", "rgb24", "bgr24", "rgba32", "argb32", "abgr32",
    "bgra32".

    ``base_pixel_scale`` is scaling factor which will be applied to the
    transformation matrix before all other transformations. It is used for
    rendering to high-resolution displays.

State functions
~~~~~~~~~~~~~~~

Saving and restoring state
^^^^^^^^^^^^^^^^^^^^^^^^^^
In addtion to the ``save_state`` and ``restore_state`` methods, it is also possible
to use a ``GraphicsContext`` instance as a context manager.

.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.save_state
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.restore_state
  :noindex:

Methods controlling state
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_fill_color
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.get_fill_color
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_stroke_color
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.get_stroke_color
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_line_width
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_line_join
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_line_cap
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_line_dash
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.linear_gradient
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.radial_gradient
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_alpha
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.get_alpha
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_antialias
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.get_antialias
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_miter_limit
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_flatness
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_image_interpolation
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.get_image_interpolation
  :noindex:


Current Transformation Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These methods control the affine transformation applied to drawing operations.
The current transformation matrix is part of the graphic state and therefore
covered by calls to ``save_state()`` and ``restore_state()``

.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.translate_ctm
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.rotate_ctm
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.scale_ctm
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.concat_ctm
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_ctm
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.get_ctm
  :noindex:


Clipping functions
~~~~~~~~~~~~~~~~~~

.. note::
   All of these functions are affected by the current transformation matrix.

.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.clip_to_rect
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.clip_to_rects
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.clip
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.even_odd_clip
  :noindex:


.. _kiva_path_functions:

Path functions
~~~~~~~~~~~~~~
The path has the concept of a "current point", which can be though of as the
pen position. Many path manipulations use the current point as a starting
position for the geometry which is added to the path.

.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.begin_path
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.close_path
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.get_empty_path
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.add_path
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.move_to
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.line_to
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.lines
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.line_set
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.rect
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.rects
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.curve_to
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.quad_curve_to
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.arc
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.arc_to
  :noindex:

.. _kiva_drawing_functions:

Drawing functions
~~~~~~~~~~~~~~~~~

.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.draw_path
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.fill_path
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.eof_fill_path
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.stroke_path
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.draw_rect
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.draw_image
  :noindex:

Enhanced drawing functions
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
   These methods are not available from every backend, so you should test for
   their presence before attempting to call them.

.. automethod:: kiva.abstract_graphics_context.EnhancedAbstractGraphicsContext.draw_marker_at_points
  :noindex:
.. automethod:: kiva.abstract_graphics_context.EnhancedAbstractGraphicsContext.draw_path_at_points
  :noindex:

Text functions
~~~~~~~~~~~~~~

.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_text_drawing_mode
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_text_matrix
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.get_text_matrix
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_text_position
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.get_text_position
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.show_text
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.show_text_at_point
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.get_text_extent
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.get_full_text_extent
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_character_spacing
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.get_character_spacing
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.select_font
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_font
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.get_font
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.set_font_size
  :noindex:


Misc functions
~~~~~~~~~~~~~~

.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.width
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.height
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.flush
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.synchronize
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.begin_page
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.end_page
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.clear_rect
  :noindex:
.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.save
  :noindex:


Types
-----

Primitive types
~~~~~~~~~~~~~~~
The following conventions are used to describe input and output types:

color:
    Either a 3-tuple or 4-tuple. The represented color depends on the
    graphics context's pixel format.
rect:
    (origin_x, origin_y, width, height)
bool:
    an int that is 1 or 0
point_array:
    an array/sequence of length-2 arrays, e.g. ((x, y), (x2, y2),...)
rect_array:
    an array/sequence of rects ((x, y, w, h), (x2, y2, w2, h2), ...)
color_stop_array:
    an array/sequence of color stops ((offset, r, g, b, a),
    (offset2, r2, g2, b2, a2), ...) where offset is some number between 0 and 1
    inclusive and the entries are sorted from lowest offset to highest.

AffineMatrix
~~~~~~~~~~~~
All of the following member functions modify the instance on which they
are called:

* ``__init__(v0, v1, v2, v3, v4, v5)``
    also __init__()
* ``reset()``
    Sets this matrix to the identity
* ``multiply(AffineMatrix)``
    multiples this matrix by another.
* ``invert()``
    sets this matrix to the inverse of itself
* ``flip_x()``
    mirrors around X
* ``flip_y()``
    mirrors around Y

The rest of the member functions return information about the matrix.

* ``scale() -> float``
    returns the average scale of this matrix
* ``determinant() -> float``
    returns the determinant
* ``asarray() -> array``
    returns the matrix as a 1D numpy array of floats

The following factory methods are available in the top-level "agg" namespace
to create specific kinds of :class:`AffineMatrix` instances:

* ``translation_matrix(float x, float x)``
* ``rotation_matrix(float angle_in_radians)``
* ``scaling_matrix(float x_scale, float y_scale)``
* ``skewing_matrix(float x_shear, float y_shear)``

Enumerations
~~~~~~~~~~~~
The following enumerations are represented by top-level constants in the "agg"
namespace.  They are fundamentally integers.  Some of them also have dicts that
map between their names and integer values

line_cap:
    CAP_BUTT, CAP_ROUND, CAP_SQUARE
line_join:
    JOIN_ROUND, JOIN_BEVEL, JOIN_MITER
draw_mode:
    FILL, EOF_FILL, STROKE, FILL_STROKE, EOF_FILL_STROKE

text_style:
    NORMAL, BOLD, ITALIC
text_draw_mode:
    TEXT_FILL, TEXT_STROKE, TEXT_FILL_STROKE, TEXT_INVISIBLE, TEXT_FILL_CLIP,
    TEXT_STROKE_CLIP, TEXT_FILL_STROKE_CLIP, TEXT_CLIP

pix_format:
    (NOTE: the strings in the dicts omit the ``pix_format_`` prefix)

    dicts:
        pix_format_string_map, pix_format_enum_map
    values:
        pix_format_gray8, pix_format_rgb555, pix_format_rgb565,
        pix_format_rgb24, pix_format_bgr24, pix_format_rgba32, pix_format_argb32,
        pix_format_abgr32, pix_format_bgra32

interpolation:
    dicts:
        interp_enum_map, interp_string_map
    values:
        nearest, bilinear, bicubic, spline16, spline36, sinc64, sinc144,
        sinc256, blackman64, blackman100, blackman256

marker:
    (NOTE: the strings in the dicts omit the ``marker_`` prefix)

    dicts:
        marker_string_map, marker_enum_map
    values:
        marker_circle, marker_cross, marker_crossed_circle, marker_dash,
        marker_diamond, marker_dot, marker_four_rays, marker_pixel,
        marker_semiellipse_down, marker_semiellipse_left, marker_x,
        marker_semiellipse_right, marker_semiellipse_up, marker_square,
        marker_triangle_down, marker_triangle_left, marker_triangle_right,
        marker_triangle_up

path_cmd and path_flags are low-level Agg path attributes.  See the Agg
documentation for more information about them.  We just pass them through in Kiva.

path_cmd:
    path_cmd_curve3, path_cmd_curve4, path_cmd_end_poly,
    path_cmd_line_to, path_cmd_mask, path_cmd_move_to, path_cmd_stop

path_flags:
    path_flags, path_flags_ccw, path_flags_close, path_flags_cw,
    path_flags_mask, path_flags_none
