from __future__ import absolute_import

from abc import ABCMeta, abstract_method

from .constants import FILL_STROKE, marker_square


class AbstractGraphicsContext(object):
    """ Abstract Base Class for Kiva Graphics Contexts """

    __metaclass__ = ABCMeta

    #----------------------------------------------------------------
    # Save/Restore graphics state.
    #----------------------------------------------------------------

    @abstract_method
    def save_state(self):
        """ Push the current graphics state onto the stack """

    @abstract_method
    def restore_state(self):
        """ Pop the previous graphics state from the stack """

    #----------------------------------------------------------------
    # context manager interface
    #----------------------------------------------------------------

    def __enter__(self):
        self.save_state()

    def __exit__(self, type, value, traceback):
        self.restore_state()

    #-------------------------------------------
    # Graphics state methods
    #-------------------------------------------

    @abstract_method
    def set_stroke_color(self, color):
        """ Set the color used when stroking a path """

    @abstract_method
    def get_stroke_color(self):
        """ Get the current color used when stroking a path """

    @abstract_method
    def set_line_width(self, width):
        """ Set the width of the pen used to stroke a path """

    @abstract_method
    def set_line_join(self, line_join):
        """ Set the style of join to use a path corners """

    @abstract_method
    def set_line_cap(self, line_cap):
        """ Set the style of cap to use a path ends """

    @abstract_method
    def set_line_dash(self, line_dash):
        """ Set the dash style to use when stroking a path

        Parameters
        ----------
            line_dash
                An even-lengthed tuple of floats that represents
                the width of each dash and gap in the dash pattern.

        """

    @abstract_method
    def set_fill_color(self, color):
        """ Set the color used to fill the region bounded by a path """

    @abstract_method
    def get_fill_color(self):
        """ Get the color used to fill the region bounded by a path """

    @abstract_method
    def linear_gradient(self, x1, y1, x2, y2, stops, spread_method, units):
        """ Modify the fill color to be a linear gradient """

    @abstract_method
    def radial_gradient(self, cx, cy, r, fx, fy, stops, spread_method, units):
        """ Modify the fill color to be a linear gradient """

    @abstract_method
    def set_alpha(self, alpha):
        """ Set the alpha to use when drawing """

    @abstract_method
    def get_alpha(self, alpha):
        """ Return the alpha used when drawing """

    @abstract_method
    def set_antialias(self, antialias):
        """ Set whether or not to antialias when drawing """

    @abstract_method
    def get_antialias(self):
        """ Set whether or not to antialias when drawing """

    @abstract_method
    def set_miter_limit(self, miter_limit):
        """ Set the limit at which mitered joins are flattened """

    @abstract_method
    def set_flatness(self, flatness):
        """ Set the error tolerance when drawing curved paths """

    @abstract_method
    def set_image_interpolation(self, interpolation):
        """ Set the type of interpolation to use when scaling images """

    @abstract_method
    def get_image_interpolation(self):
        """ Get the type of interpolation to use when scaling images """

    #-------------------------------------------
    # Transformation matrix methods
    #-------------------------------------------

    def translate_ctm(self, x, y):
        """ Concatenate a translation to the current transformation matrix """

    def rotate_ctm(self, angle):
        """ Concatenate a rotation to the current transformation matrix """

    def concat_ctm(self, matrix):
        """ Concatenate an arbitrary affine matrix to the current
            transformation matrix """

    def scale_ctm(self, x_scale, y_scale):
        """ Concatenate a scaling to the current transformation matrix """

    def set_ctm(self, matrix):
        """ Set the current transformation matrix """

    def get_ctm(self):
        """ Get the current transformation matrix """

    #-------------------------------------------
    # Clipping functions
    #-------------------------------------------

    @abstract_method
    def clip_to_rect(self, rect):
        """ Set the clipping region to the specified rectangle """

    @abstract_method
    def clip_to_rects(self, rect_array):
        """ Set the clipping region to the collection of rectangles """

    @abstract_method
    def clip(self):
        """ Set the clipping region to the current path """

    @abstract_method
    def even_odd_clip(self):
        """ Modify clipping region with the current path using even-odd rule """

    #-------------------------------------------
    # Path construction functions
    #-------------------------------------------

    @abstract_method
    def begin_path(self):
        """ Start a new path """

    @abstract_method
    def close_path(self):
        """ Finish a subpath, connecting back to the start """

    @abstract_method
    def get_empty_path(self):
        """ Get an empty CompiledPath instance """

    @abstract_method
    def add_path(self, compiled_path):
        """ Add the current path to a compiled path """

    @abstract_method
    def move_to(self, x, y):
        """ Move the current point on the path without drawing """

    @abstract_method
    def line_to(self, x, y):
        """ Add a line from the current point to (x, y) to the path """

    @abstract_method
    def lines(self, points):
        """ Adds a series of lines as a new subpath.

        Parameters
        ----------
        points
            an Nx2 sequence of (x, y) pairs

        The current_point is moved to the last point in `point_array`.

        """

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

    @abstract_method
    def rect(self, x, y, w, h):
        """ Add a rectangle as a new sub-path

        The bottom left corner is (x, y) the width is w and height is h.

        """

    @abstract_method
    def rects(self, rect_array):
        """ Add a sequence of rectangles as separate sub-paths.

        Parameters
        ----------
        rect_array:
            An Nx4 array of (x, y, w, h) quadruples

        """


    @abstract_method
    def curve_to(self, x1, y1, x2, y2, end_x, end_y):
        """ Draw a cubic bezier curve

        The curve starts from the current point and ends at (end_x, end_y),
        with control points (x1,y1) and (x2,y2).

        """

    @abstract_method
    def quad_curve_to(self, cp_x, cp_y, end_x, end_y):
        """ Draw a quadratic bezier curve

        The curve starts the current point and ends at (end_x, end_y),
        with control point (cp_x, cp_y)

        """
    @abstract_method
    def arc(self, x, y, radius, start_angle, end_angle, cw=False):
        """ Draw a circular arc of the given radius, centered at (x,y)

        The angular span is from start_angle to end_angle, where angles are
        measured counter-clockwise from the positive X axis.

        If "cw" is true, then the arc is swept from the end_angle back to the
        start_angle (it does not change the sense in which the angles are
        measured, but may affect rendering based on winding number
        calculations).

        """

    @abstract_method
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

    #-------------------------------------------
    # Drawing functions
    #-------------------------------------------

    @abstract_method
    def stroke_path(self):
        """ Stroke the current path with pen settings from current state """

    @abstract_method
    def fill_path(self):
        """ Fill the current path with fill settings from the current state

        This fills using the nonzero rule filling algorithm

        """

    @abstract_method
    def eof_fill_path(self):
        """ Fill the current path with fill settings from the current state

        This fills using the even-odd rule filling algorithm

        """

    @abstract_method
    def draw_path(self, draw_mode=FILL_STROKE):
        """ Draw the current path with the specified mode """

    @abstract_method
    def draw_rect(self, rect, draw_mode=FILL_STROKE):
        """ Draw a rectangle with the specified mode

        The rectangle is specified by a tuple (x, y, w, h).

        """

    @abstract_method
    def draw_marker_at_points(self, point_array, size, marker=marker_square):
        """ Draw a marker at a collection of points

        The shape and size of the marker are specified by the size and marker
        arguments.

        """

    @abstract_method
    def draw_path_at_points(self, point_array, compiled_path, draw_mode):
        """ Draw a compiled path at a collection of points

        The starting point of the paths are specified by the points,
        and the drawing mode is specified by the third argument.

        """

    @abstract_method
    def draw_image(image, rect=None):
        """ Render an image into a rectangle

        The rectangle is specified as an (x, y, w, h) tuple.  If it is not
        specified then the bounds of the the graphics context are used as
        the rectangle.

        """

    #-------------------------------------------
    # Text functions
    #-------------------------------------------

    @abstract_method
    def set_text_drawing_mode(self, draw_mode):
        """ Set the drawing mode to use with text """

    @abstract_method
    def set_text_matrix(self, text_matrix):
        """ Set the transformation matrix to use when drawing text """

    @abstract_method
    def get_text_matrix(self):
        """ Get the transformation matrix to use when drawing text """

    @abstract_method
    def set_text_position(self, x, y):
        """ Set the current point for drawing text

        This point is on the baseline of the text

        """

    @abstract_method
    def get_text_position(self):
        """ Get the current point where text will be drawn """

    @abstract_method
    def show_text(self, text):
        """ Draw the specified string at the current point """

    @abstract_method
    def show_text_translate(self, text, dx, dy):
        """ Draw the specified text translated as specified """

    @abstract_method
    def get_text_extent(self, text):
        """ Return a rectangle which encloses the specified text

        The rectangle (x, y, w, h) is relative to an origin which is at the
        baseline of the text and at the left of the first character rendered.
        In other words, x is the leading and y the descent.

        """

    @abstract_method
    def get_full_text_extent(self, string):
        """ Get the text extent as a tuple (w, h, x, y)

        This method is deprecated: you should use get_text_extent() instead.
        This order is provided for backwards-compatibility with existing
        Enable code.

        """

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

    def set_font(self, font):
        """ Set the font with a Kiva font object """

    def get_font(self):
        """ Get the current font """

    def set_font_size(self, size):
        """ Set the size of the current font """

    def set_character_spacing(self, spacing):
        """ Set the spacing between characters when drawing text

        Parameters
        ----------

        spacing : float
            units of space extra space to add between text coordinates.
            It is specified in text coordinate system.

        """

    def get_character_spacing(self):
        """ Get the current spacing between characters when drawing text """

    def show_text_at_point(self, x, y):
        """ Draw text at the absolute position specified by the point """

    #-------------------------------------------
    # Misc functions
    #-------------------------------------------

    def width(self):
        """ Get the width of the context manager """

    def height(self):
        """ Get the height of the context manager """

    def stride(self):
        """ Get the stride of the context manager """

    def bottom_up(self):
        """ Whether the origin is top left as opposed to bottom left """

    def format(self):
        """ The format of the color information in pixel data structures """

    def flush(self):
        """ Render all pending draw operations immediately

        This only makes sense in GUI window contexts (eg. Quartz or QPainter).

        """

    def synchronize(self):
        """ A deferred version of flush()

        Also only relevant in window contexts.

        """

    def begin_page(self):
        """ Start rendering in a new page """

    def end_page(self):
        """ Finish rendering in a page """

    def clear_rect(self, rect):
        """ Set rectangle to background colour

        This may not be available in some backends, such as PDF or PostScript.

        """

    def convert_pixel_format(self, pix_format, inplace=False):
        """ Change the way pixel data is stored.

        If inplace is True, it will try to re-use the memory that is currently
        used for the context.  This method only makes sense in rastering
        contexts.

        """

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
