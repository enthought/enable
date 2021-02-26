Kiva Interface Quick Reference
==============================

This document is a summary of the classes and functions available in
Kiva.  More specifically, it describes some of the details of the
kiva.agg backend, including enumerated types and helper classes.

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

AggFontType
~~~~~~~~~~~
This is an internal representation of fonts for the ``kiva.agg`` backend. Use
:class:`Font` instead.

* ``__init__(name, size=12, family=0, style=0)``
    constructs a :class:`AggFontType` instance
* ``is_loaded() -> bool``
    returns True if a font was actually loaded

CompiledPath
~~~~~~~~~~~~
Interface is the same as the `Path functions`_ in Graphics Context.

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


Graphics Context
----------------

Construction
~~~~~~~~~~~~
__init__(ary_or_size, pix_format="bgra32", interpolation="nearest", base_pixel_scale=1.0):
    ``ary_or_size`` can be either a numpy array or a tuple of the form
    (width, height). If it is an array, it will be used as the backing store
    for the pixels. **Its shape must be compatible with ``pix_format``**

    ``pix_format`` determines the pixel format and is a string which can be any
    of the following: "gray8", "rgb555", "rgb565", "rgb24", "bgr24", "rgba32",
    "argb32", "abgr32", "bgra32".

    ``interpolation`` determines the interpolation used by scaled image drawing
    and is a string which can be any of the following: "nearest", "bilinear",
    "bicubic", "spline16", "spline36", "sinc64", "sinc144", "sinc256",
    "blackman64", "blackman100", "blackman256".

    ``base_pixel_scale`` is scaling factor which will be applied to the
    transformation matrix before all other transformations. It is used for
    rendering to high-resolution displays.

State functions
~~~~~~~~~~~~~~~
* ``save_state()``
* ``restore_state()``
* ``set_stroke_color(color)``
* ``get_stroke_color() -> color``
* ``set_line_width(float)``
* ``set_line_join(line_join)``
* ``set_line_cap(line_cap)``
* ``set_line_dash(array)``

    ``array`` is an even-length tuple of floats that represents the width of
    each dash and gap in the dash pattern.

* ``set_fill_color(color)``
* ``get_fill_color() -> color``
* ``linear_gradient(x1, y1, x2, y2, color_stop_array, spread_method, units)``
    This method modifies the current fill pattern.

    ``spread_method`` is one of the following strings: "pad", "reflect",
    "repeat".

    ``units`` is one of the following strings: "userSpaceOnUse",
    "objectBoundingBox".
* ``radial_gradient(cx, cy, r, fx, fy, color_stop_array, spread_method, units)``
    same arguments as ``linear_gradient``. The direction of the gradient is
    from the focus point to the center point.
* ``set_alpha(float)``
* ``get_alpha() -> float``
* ``set_antialias(bool)``
* ``get_antialias() -> bool``
* ``set_miter_limit(float)``
* ``set_flatness(float)``
* ``get_image_interpolation() -> interpolation``
* ``set_image_interpolation(interpolation)``

* ``translate_ctm(float x, float y)``
* ``rotate_ctm(float angle_in_radians)``
* ``concat_ctm(AffineMatrix)``
* ``scale_ctm(float x_scale, float y_scale)``
* ``set_ctm(AffineMatrix)``
* ``get_ctm() -> AffineMatrix``


Clipping functions
~~~~~~~~~~~~~~~~~~
* ``clip_to_rect(rect)``
* ``clip_to_rects(rect_array)``
* ``clip()``
    clips using the current path
* ``even_odd_clip()``
    modifies the current clipping path using the even-odd rule to
    calculate the intersection of the current path and the current clipping path.

Path functions
~~~~~~~~~~~~~~
* ``begin_path()``
* ``close_path()``
* ``get_empty_path() -> CompiledPath``
    returns a blank :class:`CompiledPath` instance
* ``add_path(CompiledPath)``
* ``move_to(x, y)``
* ``line_to(x, y)``
* ``lines(point_array)``
* ``rect(x, y, w, h)``
* ``rects(rect_array)``
* ``curve_to(x1, y1, x2, y2, end_x, end_y)``
    draws a cubic bezier curve with
    control points (x1, y1) and (x2, y2) that ends at point (end_x, end_y)

* ``quad_curve_to(cp_x, cp_y, end_x, end_y)``
    draws a quadratic bezier curve from the current point using control point
    (cp_x, cp_y) and ending at (end_x, end_y)

* ``arc(x, y, radius, start_angle, end_angle, bool cw=false)``
    draws a circular arc of the given radius, centered at (x,y) with angular
    span as indicated.

    Angles are measured counter-clockwise from the positive X axis. If "cw" is
    true, then the arc is swept from the end_angle back to the start_angle
    (it does not change the sense in which the angles are measured).

* ``arc_to(x1, y1, x2, y2, radius)``
    Sweeps a circular arc from the pen position to a point on the line from
    (x1, y1) to (x2, y2).

    The arc is tangent to the line from the current pen position
    to (x1, y1), and it is also tangent to the line from (x1, y1)
    to (x2, y2). (x1, y1) is the imaginary intersection point of
    the two lines tangent to the arc at the current point and
    at (x2, y2).

    If the tangent point on the line from the current pen position
    to (x1, y1) is not equal to the current pen position, a line is
    drawn to it. Depending on the supplied radius, the tangent
    point on the line fron (x1, y1) to (x2, y2) may or may not be
    (x2, y2). In either case, the arc is drawn to the point of
    tangency, which is also the new pen position.

    Consider the common case of rounding a rectangle's upper left
    corner. Let "r" be the radius of rounding. Let the current
    pen position be (x_left + r, y_top). Then (x2, y2) would be
    (x_left, y_top - radius), and (x1, y1) would be (x_left, y_top).

Drawing functions
~~~~~~~~~~~~~~~~~
* ``stroke_path()``
* ``fill_path()``
* ``eof_fill_path()``
* ``draw_path(draw_mode=FILL_STROKE)``
* ``draw_rect(rect, draw_mode=FILL_STROKE)``
* ``draw_marker_at_points(point_array, int size, marker=marker_square)``
* ``draw_path_at_points(point_array, CompiledPath, draw_mode)``
* ``draw_image(graphics_context img, rect=None)``
    If ``rect`` is defined, then ``img`` is scaled and drawn into it. Otherwise,
    ``img`` is overlaid exactly on top of this graphics context.

Text functions
~~~~~~~~~~~~~~
* ``set_text_drawing_mode(text_draw_mode)``
* ``set_text_matrix(AffineMatrix)``
* ``get_text_matrix() -> AffineMatrix``
* ``set_text_position(float x, float x)``
* ``get_text_position() -> (x, y)``
* ``show_text(string)``
* ``show_text_translate(string, float y, float y)``
* ``get_text_extent(string) -> (x, y, w, h)``
* ``get_full_text_extent(string) -> (w, h, x, y)``
    deprecated. Order has been changed for backwards-compatibility with
    existing Enable.
* ``select_font(name, size, style)``
* ``set_font(AggFontType)``
* ``get_font() -> AggFontType``
* ``set_font_size(int)``
* ``set_character_spacing()``
* ``get_character_spacing()``
* ``set_text_drawing_mode()``
* ``show_text_at_point()``

Misc functions
~~~~~~~~~~~~~~
* ``width() -> int``
* ``height() -> int``
* ``stride() -> int``
* ``bottom_up() -> bool``
* ``format() -> pix_format``
* ``flush()``
    Force all pending drawing operations to be rendered immediately. This
    only makes sense in window contexts, ie- the Mac Quartz backend.
* ``synchronize()``
    A deferred version of flush(). Also only relevant in window contexts.
* ``begin_page()``
* ``end_page()``
* ``clear_rect(rect)``
    Clears a rect. Not available in PDF context
* ``convert_pixel_format(pix_format, bool inplace=0)``
* ``save(filename, file_format=None, pil_options=None)``
    Save the GraphicsContext to a file. Output files are always saved in RGB
    or RGBA format; if this GC is not in one of these formats, it is
    automatically converted.

    If ``filename`` includes an extension, the image format is
    inferred from it. ``file_format`` is only required if the
    format can't be inferred from the filename (e.g. if you
    wanted to save a PNG file as a .dat or .bin).

    ``pil_options`` is a dict of format-specific options that
    are passed down to the PIL image file writer. If a writer
    doesn't recognize an option, it is silently ignored.

    If the image has an alpha channel and the specified output
    file format does not support alpha, the image is saved in
    rgb24 format.


Functions that are currently stubbed out or not implemented
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``show_glyphs_at_point()``
