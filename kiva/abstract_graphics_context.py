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
        """ Set the color used when stroking a path """

    @abstractmethod
    def get_stroke_color(self):
        """ Get the current color used when stroking a path """

    @abstractmethod
    def set_line_width(self, width):
        """ Set the width of the pen used to stroke a path """

    @abstractmethod
    def set_line_join(self, line_join):
        """ Set the style of join to use a path corners """

    @abstractmethod
    def set_line_cap(self, line_cap):
        """ Set the style of cap to use a path ends """

    @abstractmethod
    def set_line_dash(self, line_dash):
        """ Set the dash style to use when stroking a path

        Parameters
        ----------
            line_dash
                An even-lengthed tuple of floats that represents
                the width of each dash and gap in the dash pattern.

        """

    @abstractmethod
    def set_fill_color(self, color):
        """ Set the color used to fill the region bounded by a path """

    @abstractmethod
    def get_fill_color(self):
        """ Get the color used to fill the region bounded by a path """

    @abstractmethod
    def linear_gradient(self, x1, y1, x2, y2, stops, spread_method, units):
        """ Modify the fill color to be a linear gradient """

    @abstractmethod
    def radial_gradient(self, cx, cy, r, fx, fy, stops, spread_method, units):
        """ Modify the fill color to be a linear gradient """

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
        """ Set the limit at which mitered joins are flattened """

    @abstractmethod
    def set_flatness(self, flatness):
        """ Set the error tolerance when drawing curved paths """

    @abstractmethod
    def set_image_interpolation(self, interpolation):
        """ Set the type of interpolation to use when scaling images """

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
        """ Concatenate a rotation to the current transformation matrix """

    @abstractmethod
    def concat_ctm(self, matrix):
        """ Concatenate an arbitrary affine matrix to the current
            transformation matrix """

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
        """ Get an empty CompiledPath instance """

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

        The current_point is moved to the last point in `point_array`.

        """

    @abstractmethod
    def line_set(self, starts, ends):
        """ Adds a set of disjoint lines as a new subpath.

        Parameters
        ----------
        starts
            an Nx2 array of x,y pairs
        ends
            an Nx2 array of x,y pairs

        Starts and ends arrays should have the same length.
        The current point is moved to the last point in 'ends'.

        """

    @abstractmethod
    def rect(self, x, y, w, h):
        """ Add a rectangle as a new sub-path

        The bottom left corner is (x, y) the width is w and height is h.

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

        The curve starts from the current point and ends at (end_x, end_y),
        with control points (x1,y1) and (x2,y2).

        """

    @abstractmethod
    def quad_curve_to(self, cp_x, cp_y, end_x, end_y):
        """ Draw a quadratic bezier curve

        The curve starts the current point and ends at (end_x, end_y),
        with control point (cp_x, cp_y)

        """

    @abstractmethod
    def arc(self, x, y, radius, start_angle, end_angle, cw=False):
        """ Draw a circular arc of the given radius, centered at (x,y)

        The angular span is from start_angle to end_angle, where angles are
        measured counter-clockwise from the positive X axis.

        If "cw" is true, then the arc is swept from the end_angle back to the
        start_angle (it does not change the sense in which the angles are
        measured, but may affect rendering based on winding number
        calculations).

        """

    @abstractmethod
    def arc_to(self, x1, y1, x2, y2, radius):
        """ Draw a circular arc from current point to tangent line

        The arc is tangent to the line from the current pen position
        to (x1,y1), and it is also tangent to the line from (x1,y1)
        to (x2,y2).  (x1,y1) is the imaginary intersection point of
        the two lines tangent to the arc at the current point and
        at (x2,y2).

        If the tangent point on the line from the current pen position
        to (x1,y1) is not equal to the current pen position, a line is
        drawn to it.  Depending on the supplied radius, the tangent
        point on the line fron (x1,y1) to (x2,y2) may or may not be
        (x2,y2).  In either case, the arc is drawn to the point of
        tangency, which is also the new pen position.

        Consider the common case of rounding a rectangle's upper left
        corner.  Let "r" be the radius of rounding.  Let the current
        pen position be (x_left + r, y_top).  Then (x2,y2) would be
        (x_left, y_top - radius), and (x1,y1) would be (x_left, y_top).

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
        """ Draw the current path with the specified mode """

    @abstractmethod
    def draw_rect(self, rect, draw_mode=FILL_STROKE):
        """ Draw a rectangle with the specified mode

        The rectangle is specified by a tuple (x, y, w, h).

        """

    @abstractmethod
    def draw_image(self, image, rect=None):
        """ Render an image into a rectangle

        The rectangle is specified as an (x, y, w, h) tuple.  If it is not
        specified then the bounds of the the graphics context are used as
        the rectangle.

        """

    # -------------------------------------------
    # Text functions
    # -------------------------------------------

    @abstractmethod
    def set_text_drawing_mode(self, draw_mode):
        """ Set the drawing mode to use with text """

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
    def show_text(self, text):
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

        This method is deprecated: you should use get_text_extent() instead.
        This order is provided for backwards-compatibility with existing
        Enable code.

        """

    @abstractmethod
    def select_font(self, name, size=12, style="regular", encoding=None):
        """ Set the font based on the provided parameters

        Parameters
        ----------

        name:
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
        """ Set the font with a Kiva font object """

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

        This may not be available in some backends, such as PDF or PostScript.

        """

    @abstractmethod
    def save(self, filename, file_format=None, pil_options=None):
        """ Save the graphics context to a file

        Data is always saved in RGB or RGBA format, and converted to that
        format if not already in it.

        If the file_format argument is None, then the file format is inferred
        from the filename extension, and so is not usually needed.

        The pil_options argument is a dictionary of format-specific options
        that can be passed directly to PIL's image file writers.  For example,
        this can be used to control the compression level of JPEG or PNG
        output.  Unrecognized options are silently ignored.

        """


class EnhancedAbstractGraphicsContext(AbstractGraphicsContext):
    """ ABC for graphics contexts which provide additional methods """

    @abstractmethod
    def draw_marker_at_points(self, point_array, size, marker=SQUARE_MARKER):
        """ Draw a marker at a collection of points

        The shape and size of the marker are specified by the size and marker
        arguments.

        """

    @abstractmethod
    def draw_path_at_points(self, point_array, compiled_path, draw_mode):
        """ Draw a compiled path at a collection of points

        The starting point of the paths are specified by the points,
        and the drawing mode is specified by the third argument.

        """

    @abstractmethod
    def show_text_translate(self, text, dx, dy):
        """ Draw the specified text translated as specified """
