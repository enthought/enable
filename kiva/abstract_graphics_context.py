# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from abc import ABCMeta, abstractmethod

from .constants import FILL_STROKE, SQUARE_MARKER


class AbstractGraphicsContext(object, metaclass=ABCMeta):
    """ Abstract Base Class for Kiva Graphics Contexts """

    # ----------------------------------------------------------------
    # Save/Restore graphics state.
    # ----------------------------------------------------------------

    @abstractmethod
    def save_state(self):
        """ Push the current graphics state onto the stack """

    @abstractmethod
    def restore_state(self):
        """ Pop the previous graphics state from the stack """

    # ----------------------------------------------------------------
    # context manager interface
    # ----------------------------------------------------------------

    def __enter__(self):
        self.save_state()

    def __exit__(self, type, value, traceback):
        self.restore_state()

    # -------------------------------------------
    # Graphics state methods
    # -------------------------------------------

    @abstractmethod
    def set_stroke_color(self, color):
        """ Set the color used when stroking a path

        Parameters
        ----------
            color
                A three or four component tuple describing a color
                (R, G, B[, A]). Each color component should be in the range
                [0.0, 1.0]
        """

    @abstractmethod
    def get_stroke_color(self):
        """ Get the current color used when stroking a path """

    @abstractmethod
    def set_line_width(self, width):
        """ Set the width of the pen used to stroke a path """

    @abstractmethod
    def set_line_join(self, line_join):
        """ Set the style of join to use a path corners

        Parameters
        ----------
            line_join
                Options are ``JOIN_ROUND``, ``JOIN_BEVEL``, or ``JOIN_MITER``.
                Each is defined in :py:mod:`kiva.api`.
        """

    @abstractmethod
    def set_line_cap(self, line_cap):
        """ Set the style of cap to use a path ends

        Parameters
        ----------
            line_cap
                One of ``CAP_BUTT``, ``CAP_ROUND``, or ``CAP_SQUARE``.
                Each is defined in :py:mod:`kiva.api`.
        """

    @abstractmethod
    def set_line_dash(self, line_dash, phase=0):
        """ Set the dash style to use when stroking a path

        Parameters
        ----------
            line_dash
                An even-lengthed tuple of floats that represents
                the width of each dash and gap in the dash pattern.
            phase : float
                Specifies how many units into the dash pattern to start.
        """

    @abstractmethod
    def set_fill_color(self, color):
        """ Set the color used to fill the region bounded by a path or when
        drawing text.

        Parameters
        ----------
            color
                A three or four component tuple describing a color
                (R, G, B[, A]). Each color component should be in the range
                [0.0, 1.0]
        """

    @abstractmethod
    def get_fill_color(self):
        """ Get the color used to fill the region bounded by a path or when
        drawing text.
        """

    @abstractmethod
    def linear_gradient(self, x1, y1, x2, y2, stops, spread_method,
                        units="userSpaceOnUse"):
        """ Modify the fill color to be a linear gradient

        Parameters
        ----------
            x1
                The X starting point of the gradient.
            y1
                The Y starting point of the gradient.
            x2
                The X ending point of the gradient.
            y3
                The Y ending point of the gradient.
            stops
                An array/sequence of color stops
                ((offset, r, g, b, a), (offset2, r2, g2, b2, a2), …) where
                offset is some number between 0 and 1 inclusive and the entries
                are sorted from lowest offset to highest.
            spread_method
                One of the following strings: "pad", "reflect", "repeat".
            units
                One of the following strings: "userSpaceOnUse",
                "objectBoundingBox".
        """

    @abstractmethod
    def radial_gradient(self, cx, cy, r, fx, fy, stops, spread_method,
                        units="userSpaceOnUse"):
        """ Modify the fill color to be a radial gradient

        Parameters
        ----------
            cx
                The X center point of the gradient.
            cy
                The Y center point of the gradient.
            r
                The radius of the gradient
            fx
                The X ending point of the gradient.
            fy
                The Y ending point of the gradient.
            stops
                An array/sequence of color stops
                ((offset, r, g, b, a), (offset2, r2, g2, b2, a2), …) where
                offset is some number between 0 and 1 inclusive and the entries
                are sorted from lowest offset to highest.
            spread_method
                One of the following strings: "pad", "reflect", "repeat".
            units
                One of the following strings: "userSpaceOnUse",
                "objectBoundingBox".
        """

    @abstractmethod
    def set_alpha(self, alpha):
        """ Set the alpha to use when drawing """

    @abstractmethod
    def get_alpha(self, alpha):
        """ Return the alpha used when drawing """

    @abstractmethod
    def set_antialias(self, antialias):
        """ Set whether or not to antialias when drawing """

    @abstractmethod
    def get_antialias(self):
        """ Set whether or not to antialias when drawing """

    @abstractmethod
    def set_miter_limit(self, miter_limit):
        """ Set the limit at which mitered joins are flattened.

        Only applicable when the line join type is set to ``JOIN_MITER``.
        """

    @abstractmethod
    def set_flatness(self, flatness):
        """ Set the error tolerance when drawing curved paths """

    @abstractmethod
    def set_image_interpolation(self, interpolation):
        """ Set the type of interpolation to use when scaling images.

        Parameters
        ----------
            interpolation
                One of "nearest", "bilinear", "bicubic", "spline16",
                "spline36", "sinc64", "sinc144", "sinc256", "blackman64",
                "blackman100", or "blackman256".
        """

    @abstractmethod
    def get_image_interpolation(self):
        """ Get the type of interpolation to use when scaling images """

    # -------------------------------------------
    # Transformation matrix methods
    # -------------------------------------------

    @abstractmethod
    def translate_ctm(self, x, y):
        """ Concatenate a translation to the current transformation matrix """

    @abstractmethod
    def rotate_ctm(self, angle):
        """ Concatenate a rotation to the current transformation matrix.

        Parameters
        ----------
            angle
                An angle in radians.
        """

    @abstractmethod
    def concat_ctm(self, matrix):
        """ Concatenate an arbitrary affine matrix to the current
        transformation matrix.
        """

    @abstractmethod
    def scale_ctm(self, x_scale, y_scale):
        """ Concatenate a scaling to the current transformation matrix """

    @abstractmethod
    def set_ctm(self, matrix):
        """ Set the current transformation matrix """

    @abstractmethod
    def get_ctm(self):
        """ Get the current transformation matrix """

    # -------------------------------------------
    # Clipping functions
    # -------------------------------------------

    @abstractmethod
    def clip_to_rect(self, rect):
        """ Set the clipping region to the specified rectangle """

    @abstractmethod
    def clip_to_rects(self, rect_array):
        """ Set the clipping region to the collection of rectangles """

    @abstractmethod
    def clip(self):
        """ Set the clipping region to the current path """

    @abstractmethod
    def even_odd_clip(self):
        """ Modify clipping region with current path using even-odd rule """

    # -------------------------------------------
    # Path construction functions
    # -------------------------------------------

    @abstractmethod
    def begin_path(self):
        """ Start a new path """

    @abstractmethod
    def close_path(self):
        """ Finish a subpath, connecting back to the start """

    @abstractmethod
    def get_empty_path(self):
        """ Get an empty ``CompiledPath`` instance """

    @abstractmethod
    def add_path(self, compiled_path):
        """ Add the current path to a compiled path """

    @abstractmethod
    def move_to(self, x, y):
        """ Move the current point on the path without drawing """

    @abstractmethod
    def line_to(self, x, y):
        """ Add a line from the current point to (x, y) to the path """

    @abstractmethod
    def lines(self, points):
        """ Adds a series of lines as a new subpath.

        Parameters
        ----------
            points
                an Nx2 sequence of (x, y) pairs

        The current point is moved to the last point in ``points``.
        """

    @abstractmethod
    def line_set(self, starts, ends):
        """ Adds a set of disjoint lines as a new subpath.

        Parameters
        ----------
            starts:
                an Nx2 array of x,y pairs
            ends:
                an Nx2 array of x,y pairs

        ``starts`` and ``ends`` arrays should have the same length.
        The current point is moved to the last point in ``ends``.
        """

    @abstractmethod
    def rect(self, x, y, w, h):
        """ Add a rectangle as a new sub-path

        Parameters
        ----------
            x:
                The left X coordinate of the rectangle
            y:
                The bottom Y coordinate of the rectangle
            w:
                The width of the rectangle
            h:
                The height of the rectangle
        """

    @abstractmethod
    def rects(self, rect_array):
        """ Add a sequence of rectangles as separate sub-paths.

        Parameters
        ----------
            rect_array:
                An Nx4 array of (x, y, w, h) quadruples
        """

    @abstractmethod
    def curve_to(self, x1, y1, x2, y2, end_x, end_y):
        """ Draw a cubic bezier curve

        The curve starts from the current point and ends at ``(end_x, end_y)``,
        with control points ``(x1, y1)`` and ``(x2, y2)``.
        """

    @abstractmethod
    def quad_curve_to(self, cp_x, cp_y, end_x, end_y):
        """ Draw a quadratic bezier curve

        The curve starts the current point and ends at ``(end_x, end_y)``,
        with control point ``(cp_x, cp_y)``
        """

    @abstractmethod
    def arc(self, x, y, radius, start_angle, end_angle, cw=False):
        """ Draw a circular arc of the given radius, centered at ``(x, y)``

        The angular span is from ``start_angle`` to ``end_angle``, where angles
        are measured counter-clockwise from the positive X axis.

        If "cw" is True, then the arc is swept from the ``end_angle`` back to
        the ``start_angle`` (it does not change the sense in which the angles
        are measured, but may affect rendering based on winding number
        calculations).
        """

    @abstractmethod
    def arc_to(self, x1, y1, x2, y2, radius):
        """ Draw a circular arc from current point to tangent line

        The arc is tangent to the line from the current point to ``(x1, y1)``,
        and it is also tangent to the line from ``(x1, y1)`` to ``(x2, y2)``.
        ``(x1, y1)`` is the imaginary intersection point of the two lines
        tangent to the arc at the current point and at ``(x2, y2)``.

        If the tangent point on the line from the current point to ``(x1, y1)``
        is not equal to the current point, a line is drawn to it. Depending on
        the supplied ``radius``, the tangent point on the line from
        ``(x1, y1)`` to ``(x2, y2)`` may or may not be ``(x2, y2)``. In either
        case, the arc is drawn to the point of tangency, which is also the new
        current point.

        Consider the common case of rounding a rectangle's upper left corner.
        Let "r" be the radius of rounding. Let the current point be
        ``(x_left + r, y_top)``. Then ``(x2, y2)`` would be
        ``(x_left, y_top - radius)``, and ``(x1, y1)`` would be
        ``(x_left, y_top)``.
        """

    # -------------------------------------------
    # Drawing functions
    # -------------------------------------------

    @abstractmethod
    def stroke_path(self):
        """ Stroke the current path with pen settings from current state """

    @abstractmethod
    def fill_path(self):
        """ Fill the current path with fill settings from the current state

        This fills using the nonzero rule filling algorithm
        """

    @abstractmethod
    def eof_fill_path(self):
        """ Fill the current path with fill settings from the current state

        This fills using the even-odd rule filling algorithm
        """

    @abstractmethod
    def draw_path(self, draw_mode=FILL_STROKE):
        """ Draw the current path with the specified mode

        Parameters
        ----------
            draw_mode
                One of ``FILL``, ``EOF_FILL``, ``STROKE``, ``FILL_STROKE``,
                or ``EOF_FILL_STROKE``. Each is defined in :py:mod:`kiva.api`.
        """

    @abstractmethod
    def draw_rect(self, rect, draw_mode=FILL_STROKE):
        """ Draw a rectangle with the specified mode

        Parameters
        ----------
            rect
                A tuple (x, y, w, h)
            draw_mode
                One of ``FILL``, ``EOF_FILL``, ``STROKE``, ``FILL_STROKE``, or
                ``EOF_FILL_STROKE``. Each is defined in :py:mod:`kiva.api`.
        """

    @abstractmethod
    def draw_image(self, image, rect=None):
        """ Render an image into a rectangle

        Parameters
        ----------
            image
                An image. Can be a numpy array, a PIL ``Image`` instance, or
                another ``GraphicsContext`` instance.
            rect
                A tuple (x, y, w, h). If not specified then the bounds of the
                the graphics context are used as the rectangle.
        """

    # -------------------------------------------
    # Text functions
    # -------------------------------------------

    @abstractmethod
    def set_text_drawing_mode(self, draw_mode):
        """ Set the drawing mode to use with text

        Parameters
        ----------
            draw_mode
                Allowed values are ``TEXT_FILL``, ``TEXT_STROKE``,
                ``TEXT_FILL_STROKE``, ``TEXT_INVISIBLE``, ``TEXT_FILL_CLIP``,
                ``TEXT_STROKE_CLIP``, ``TEXT_FILL_STROKE_CLIP``, or
                ``TEXT_CLIP``. Each is defined in :py:mod:`kiva.api`.
        """

    @abstractmethod
    def set_text_matrix(self, text_matrix):
        """ Set the transformation matrix to use when drawing text """

    @abstractmethod
    def get_text_matrix(self):
        """ Get the transformation matrix to use when drawing text """

    @abstractmethod
    def set_text_position(self, x, y):
        """ Set the current point for drawing text

        This point is on the baseline of the text
        """

    @abstractmethod
    def get_text_position(self):
        """ Get the current point where text will be drawn """

    @abstractmethod
    def show_text(self, text, point=None):
        """ Draw the specified string at the current point """

    @abstractmethod
    def get_text_extent(self, text):
        """ Return a rectangle which encloses the specified text

        The rectangle (x, y, w, h) is relative to an origin which is at the
        baseline of the text and at the left of the first character rendered.
        In other words, x is the leading and y the descent.
        """

    @abstractmethod
    def get_full_text_extent(self, string):
        """ Get the text extent as a tuple (w, h, x, y)

        .. note::
           This method is deprecated: you should use ``get_text_extent()``
           instead. This order is provided for backwards-compatibility with
           existing Enable code.
        """

    @abstractmethod
    def select_font(self, name, size=12, style="regular", encoding=None):
        """ Set the font based on the provided parameters

        Parameters
        ----------
        name
            The name of a font. E.g.: "Times New Roman"
        size
            The font size in points.
        style
            One of "regular", "bold", "italic", "bold italic"
        encoding
            A 4 letter encoding name. Common ones are:

                * "unic" -- unicode
                * "armn" -- apple roman
                * "symb" -- symbol

            Not all fonts support all encodings.  If none is
            specified, fonts that have unicode encodings
            default to unicode.  Symbol is the second choice.
            If neither are available, the encoding defaults
            to the first one returned in the FreeType charmap
            list for the font face.
        """

    @abstractmethod
    def set_font(self, font):
        """ Set the font with a :class:`kiva.fonttools.font.Font` object
        """

    @abstractmethod
    def get_font(self):
        """ Get the current font """

    @abstractmethod
    def set_font_size(self, size):
        """ Set the size of the current font """

    @abstractmethod
    def set_character_spacing(self, spacing):
        """ Set the spacing between characters when drawing text

        Parameters
        ----------
        spacing : float
            units of space extra space to add between text coordinates.
            It is specified in text coordinate system.
        """

    @abstractmethod
    def get_character_spacing(self):
        """ Get the current spacing between characters when drawing text """

    @abstractmethod
    def show_text_at_point(self, x, y):
        """ Draw text at the absolute position specified by the point """

    # -------------------------------------------
    # Misc functions
    # -------------------------------------------

    @abstractmethod
    def width(self):
        """ Returns the width of the graphics context. """

    @abstractmethod
    def height(self):
        """ Returns the height of the graphics context. """

    @abstractmethod
    def flush(self):
        """ Render all pending draw operations immediately

        This only makes sense in GUI window contexts (eg. Quartz or QPainter).
        """

    @abstractmethod
    def synchronize(self):
        """ A deferred version of flush()

        Also only relevant in window contexts.
        """

    @abstractmethod
    def begin_page(self):
        """ Start rendering in a new page """

    @abstractmethod
    def end_page(self):
        """ Finish rendering in a page """

    @abstractmethod
    def clear_rect(self, rect):
        """ Set rectangle to background colour

        .. note::
            This may not be available in some backends, such as PDF or
            PostScript.
        """

    @abstractmethod
    def save(self, filename, file_format=None, pil_options=None):
        """ Save the graphics context to a file

        Data is always saved in RGB or RGBA format, and converted to that
        format if not already in it.

        If the ``file_format`` argument is None, then the file format is
        inferred from the ``filename`` extension, and so is not usually needed.

        The ``pil_options`` argument is a dictionary of format-specific options
        that can be passed directly to PIL's image file writers. For example,
        this can be used to control the compression level of JPEG or PNG
        output. Unrecognized options are silently ignored.
        """


class EnhancedAbstractGraphicsContext(AbstractGraphicsContext):
    """ ABC for graphics contexts which provide additional methods """

    @abstractmethod
    def draw_marker_at_points(self, point_array, size, marker=SQUARE_MARKER):
        """ Draw a marker at a collection of points

        Parameters
        ----------
            point_array
                An Nx2 array of x,y points
            size
                The size of the marker in points.
            marker
                One of ``NO_MARKER``, ``SQUARE_MARKER``, ``DIAMOND_MARKER``,
                ``CIRCLE_MARKER``, ``CROSSED_CIRCLE_MARKER``, ``CROSS_MARKER``,
                ``TRIANGLE_MARKER``, ``INVERTED_TRIANGLE_MARKER``,
                ``PLUS_MARKER``, ``DOT_MARKER``, or ``PIXEL_MARKER``. Each is
                defined in :py:mod:`kiva.api`.
        """

    @abstractmethod
    def draw_path_at_points(self, point_array, compiled_path,
                            draw_mode=FILL_STROKE):
        """ Draw a compiled path at a collection of points

        The starting point of the paths are specified by the points,
        and the drawing mode is specified by the third argument.

        Parameters
        ----------
            point_array
                An Nx2 array of x,y points
            compiled_path
                A ``CompiledPath`` instance.
            draw_mode
                One of ``FILL``, ``EOF_FILL``, ``STROKE``, ``FILL_STROKE``, or
                ``EOF_FILL_STROKE``. Each is defined in :py:mod:`kiva.api`.
        """
