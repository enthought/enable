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
A path is a colection of geometry that can be draw in a graphics context with
coloring and an affine transformation applied to it. It is the basic unit of
drawing in a graphics context.

Interface is the same as the `Path functions`_ .

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
    Saves the state of the graphics context and pushes it onto a state stack.
* ``restore_state()``
    Pops the state stack, restoring the state from the previous call to
    ``save_state()``.
* ``set_fill_color(color)``
    Sets the color used when calling ``fill_path()`` or ``draw_path()`` with any
    mode which fills.
* ``get_fill_color() -> color``
    Returns the current fill color.
* ``set_stroke_color(color)``
    Sets the color used when calling ``stroke_path()`` or ``draw_path()`` with
    any mode which strokes.
* ``get_stroke_color() -> color``
    Returns the current stroke color.
* ``set_line_width(float)``
    Sets the width of stroked lines. Note that this can be affected by the
    current transformation matrix.
* ``set_line_join(line_join)``
    Sets the join type for multi-segment lines. Allowed values are
    ``JOIN_ROUND``, ``JOIN_BEVEL``, or ``JOIN_MITER``.
* ``set_line_cap(line_cap)``
    Sets the cap type for line ends. Allowed values are ``CAP_BUTT``,
    ``CAP_ROUND``, ``CAP_SQUARE``
* ``set_line_dash(array)``
    ``array`` is an even-length tuple of floats that represents the width of
    each dash and gap in the dash pattern.
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
    Sets the transparency for all drawing calls.
* ``get_alpha() -> float``
    Returns the transparency used for drawing calls.
* ``set_antialias(bool)``
    Enables or disables anti-aliasing.
* ``get_antialias() -> bool``
    Returns True if anti-aliasing is enabled.
* ``set_miter_limit(float)``
    If the line join type is set to ``JOIN_MITER``, the miter limit determines
    whether the lines should be joined with a bevel instead of a miter.

    Note that this may not be implemented by all backends.
* ``set_flatness(float)``
    Controls how accurately curved paths are rendered.

    Note that this may not be implemented by all backends.
* ``get_image_interpolation() -> interpolation``
    Returns the currently set image interpolation method.
* ``set_image_interpolation(interpolation)``
    Sets the image interpolation method. Allowed values are "nearest",
    "bilinear", "bicubic", "spline16", "spline36", "sinc64", "sinc144",
    "sinc256", "blackman64", "blackman100", and "blackman256"

    Note that this may not be implemented by all backends.

Current Transformation Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These methods control the affine transformation applied to drawing operations.
The current transformation matrix is part of the graphic state and therefore
covered by calls to ``save_state()`` and ``restore_state()``

* ``translate_ctm(float x, float y)``
    Translate in X and Y
* ``rotate_ctm(float angle_in_radians)``
    Rotate by some angle
* ``concat_ctm(AffineMatrix)``
    Premultiplies the current transformation matrix by a specified affine matrix
* ``scale_ctm(float x_scale, float y_scale)``
    Scales in X and Y
* ``set_ctm(AffineMatrix)``
    Assigns a specified affine matrix to the current transformation matrix
* ``get_ctm() -> AffineMatrix``
    Returns the current transformation matrix as an ``AffineMatrix``


Clipping functions
~~~~~~~~~~~~~~~~~~

.. note::
   All of these functions are affected by the current transformation matrix.

* ``clip_to_rect(rect)``
    Clips drawing to a single rectangle
* ``clip_to_rects(rect_array)``
    Clips drawing to a collection of rectangles.
* ``clip()``
    Clips using the current path
* ``even_odd_clip()``
    Modifies the current clipping path using the even-odd rule to
    calculate the intersection of the current path and the current clipping
    path.

Path functions
~~~~~~~~~~~~~~
The path has the concept of a "current point", which can be though of as the
pen position. Many path manipulations use the current point as a starting
position for the geometry which is added to the path.

* ``begin_path()``
    Initializes the path to an empty state.
* ``close_path()``
    Closes the current path. This means that the final point gets connected to
    the starting point.
* ``get_empty_path() -> CompiledPath``
    returns a blank :class:`CompiledPath` instance
* ``add_path(CompiledPath)``
    Adds a ``CompiledPath`` instance as a subpath of the current path
* ``move_to(x, y)``
    Moves the current point to (x, y).
* ``line_to(x, y)``
    Adds a line from the current point to the passed in point.
* ``lines(point_array)``
    Adds a collection of line segments to the current path.
* ``rect(x, y, w, h)``
    Adds a rectangle to the current path
* ``rects(rect_array)``
    Adds a collection of rectangles to the current path
* ``curve_to(x1, y1, x2, y2, end_x, end_y)``
    Adds a cubic bezier curve from the current point with control points
    (x1, y1) and (x2, y2) that ends at point (end_x, end_y)
* ``quad_curve_to(cp_x, cp_y, end_x, end_y)``
    Adds a quadratic bezier curve from the current point using control point
    (cp_x, cp_y) and ending at (end_x, end_y)
* ``arc(x, y, radius, start_angle, end_angle, bool cw=false)``
    Adds a circular arc of the given radius, centered at (x,y) with angular
    span as indicated.

    Angles are measured counter-clockwise from the positive X axis. If "cw" is
    true, then the arc is swept from the end_angle back to the start_angle
    (it does not change the sense in which the angles are measured).
* ``arc_to(x1, y1, x2, y2, radius)``
    Sweeps a circular arc from the current point to a point on the line from
    (x1, y1) to (x2, y2).

    The arc is tangent to the line from the current point
    to (x1, y1), and it is also tangent to the line from (x1, y1)
    to (x2, y2). (x1, y1) is the imaginary intersection point of
    the two lines tangent to the arc at the current point and
    at (x2, y2).

    If the tangent point on the line from the current point
    to (x1, y1) is not equal to the current point, a line is
    drawn to it. Depending on the supplied radius, the tangent
    point on the line fron (x1, y1) to (x2, y2) may or may not be
    (x2, y2). In either case, the arc is drawn to the point of
    tangency, which is also the new current point.

    Consider the common case of rounding a rectangle's upper left
    corner. Let "r" be the radius of rounding. Let the current point be
    (x_left + r, y_top). Then (x2, y2) would be
    (x_left, y_top - radius), and (x1, y1) would be (x_left, y_top).

Drawing functions
~~~~~~~~~~~~~~~~~
* ``stroke_path()``
    Strokes the current path
* ``fill_path()``
    Fills the current path using the zero-winding fill rule
* ``eof_fill_path()``
    Fills the current path using the even-odd fill rule
* ``draw_path(draw_mode=FILL_STROKE)``
    Draws the current path using a draw mode. Allowed modes are ``FILL``,
    ``EOF_FILL``, ``STROKE``, ``FILL_STROKE``, ``EOF_FILL_STROKE``
* ``draw_rect(rect, draw_mode=FILL_STROKE)``
    Draws a rectangle using the specified drawing mode.

    ``rect`` is a tuple of the form ``(x, y, width, height)``
* ``draw_marker_at_points(point_array, int size, marker=marker_square)``
    Draws markers at all the points in ``point_array``. Allowed markers are
    "circle", "cross", "crossed_circle", "dash", "diamond", "dot", "four_rays",
    "pixel", "semiellipse_down", "semiellipse_left", "x", "semiellipse_right",
    "semiellipse_up", "square", "triangle_down", "triangle_left",
    "triangle_right", and "triangle_up".

    Note: This is basically only supported by the ``kiva.agg`` backend. Use
    ``draw_path_at_points`` instead.
* ``draw_path_at_points(point_array, path, draw_mode)``
    Draws a ``CompiledPath`` object at each point in ``point_array`` using the
    specified drawing mode.

    Note: This is not available with all backends.
* ``draw_image(img, rect=None)``
    Draws an image. If ``rect`` is defined, then ``img`` is scaled and drawn
    into it. Otherwise, ``img`` is overlaid exactly on top of this graphics
    context.

    ``img`` can be a PIL ``Image`` instance, a numpy array, or another 
    ``GraphicsContext`` instance.

Text functions
~~~~~~~~~~~~~~
* ``set_text_drawing_mode(text_draw_mode)``
    Sets the text drawing mode. Allowed values are ``TEXT_FILL``,
    ``TEXT_STROKE``, ``TEXT_FILL_STROKE``, ``TEXT_INVISIBLE``,
    ``TEXT_FILL_CLIP``, ``TEXT_STROKE_CLIP``, ``TEXT_FILL_STROKE_CLIP``,
    ``TEXT_CLIP``
* ``set_text_matrix(AffineMatrix)``
    Sets a transformation matrix which only applies to text.

    Note: This is not uniformly implemented across all backends.
* ``get_text_matrix() -> AffineMatrix``
    Returns a previously set text transformation matrix.
* ``set_text_position(float x, float x)``
    Sets the position where text will be drawn by ``show_text``
* ``get_text_position() -> (x, y)``
    Returns the previously set text position.
* ``show_text(string, point=None)``
    Draws ``string`` at the current text position, or ``point`` if it is
    provided.
* ``show_text_at_point(string, float y, float y)``
    Draws string at the specified position
* ``get_text_extent(string) -> (x, y, w, h)``
    Returns the bounding box of ``string`` if rendered using the currently set
    font.
* ``get_full_text_extent(string) -> (w, h, x, y)``
    deprecated. Order has been changed for backwards-compatibility with
    existing Enable.
* ``select_font(name, size, style)``
    Selects a font using ``name``, ``size``, and ``style``. Note that this will
    be fulfilled on a best-effort basis. The system might not have the exact
    font which is requested.
* ``set_font(Font)``
    Sets the font using a :class:`kiva.api.Font` object
* ``get_font() -> Font``
    Returns the currently selected font as a :class:`kiva.api.Font` object
* ``set_font_size(int)``
    Sets the size of drawn text in points
* ``set_character_spacing()``
* ``get_character_spacing()``


Misc functions
~~~~~~~~~~~~~~
* ``width() -> int``
    Returns the width of the graphics context, in pixels
* ``height() -> int``
    Returns the height of the graphics context, in pixels
* ``format() -> pix_format``
    Returns the pixel format of the graphics context
* ``flush()``
    Force all pending drawing operations to be rendered immediately. This
    only makes sense in window contexts, ie- the Mac Quartz backend.
* ``synchronize()``
    A deferred version of flush(). Also only relevant in window contexts.
* ``begin_page()``
* ``end_page()``
* ``clear_rect(rect)``
    Clears a rect. Not available in the PDF backend,
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
