#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# some parts copyright Space Telescope Science Institute
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#------------------------------------------------------------------------------
""" Pure-Python reference implementation of a Kiva graphics context.

    Data Structures
    ---------------
    color
        1-D array with 4 elements.
        The array elements represent (red, green, blue, alpha)
        and are each between 0 and 1.  Alpha is a transparency
        value with 0 being fully transparent and 1 being fully
        opaque.  Many backends do not handle tranparency and
        treat any alpha value greater than 0 as fully opaque.
    transform
        currently a 3x3 array.  This is not the
        most convenient in some backends.  Mac and OpenGL
        use a 1-D 6 element array.  We need to either make
        transform a class or always use accessor functions
        to access its values. Currently, I do the latter.
"""

import affine
import copy
from numpy import alltrue, array, asarray, float64, sometrue, shape,\
     pi, concatenate
import numpy as np

from constants import *

def exactly_equal(arr1,arr2):
    return shape(arr1)==shape(arr2) and alltrue(arr1==arr2)

#--------------------------------------------------------------------
# Import and initialize freetype engine for rendering.
#
# !! Need to figure out how to set dpi intelligently
#--------------------------------------------------------------------
#from enthought import freetype
# freetype engine for text rendering.
#ft_engine = freetype.FreeType(dpi=120.0)

#--------------------------------------------------------------------
# Drawing style tests.
#
# Simple tests used by drawing methods to determine what kind of
# drawing command is supposed to be executed.
#--------------------------------------------------------------------

def is_point(tup): return tup[0] == POINT
def is_line(tup): return tup[0] == LINE

def is_dashed(dash):
    # if all the values in the dash settings are 0, then it is a solid line.
    result = 0
    if dash is not None and sometrue(asarray(dash[1]) != 0):
        result = 1
    return result

def is_fully_transparent(color):
    """ Tests a color array to see whether it is fully transparent or not.

        This is true if the alpha value (4th entry in the color array) is
        0.0.
    """
    transparent = (color[3] == 0.0)
    return transparent

def line_state_equal(line1,line2):
    """ Compares two `LineState` objects to see if they are equivalent.

        This is generally called by device-specific drawing routines
        before they stroke a path. It determines whether previously set
        line settings are equivalent to desired line settings for this
        drawing command.  If true, the routine can bypass all the
        work needed to set all the line settings of the graphics device.

        With the current Python implementation, this may not provide any
        time savings over just setting all the graphics state values.
        However, in C this could be a very fast memcmp if the C structure
        is set up correctly.

        While this could be the __cmp__ method for `LineState`, I have
        left it as a function because I think it will move to C and be
        used to compare structures.
    """

    result = 0
    #---------------------------------------------------------------------
    # line_dash is a little persnickety.  It is a 2-tuple
    # with the second entry being an array.  If the arrays are different,
    # just comparing the tuple will yield true because of how rich
    # the result from the array comparison is a non-empty array which
    # tests true.  Thus, the tuple comparison will test true even if the
    # arrays are different.  Its almost like we need a "deep compare"
    # method or something like that.
    #
    # Note: I think should be easy, but is breaking because of a bug in
    #       Numeric.  Waiting for confirmation.
    #---------------------------------------------------------------------
    dash_equal = line1.line_dash[0] == line2.line_dash[0] and \
                 exactly_equal(line1.line_dash[1], line2.line_dash[1])
    if (dash_equal                                    and
        exactly_equal(line1.line_color, line2.line_color)  and
        line1.line_width == line2.line_width          and
        line1.line_cap   == line2.line_cap            and
        line1.line_join  == line2.line_join):
        result = 1
    return result

def fill_equal(fill1,fill2):
    """ Currently fill just compares the two colors.


    """
    return alltrue(fill1 == fill2)

class LineState(object):
    """ Stores information about the current line drawing settings.

        This is split off from `GraphicsState` to make it easier to
        track line state changes.  All the methods for setting
        these variables are left in the GraphicsStateBase class.
    """
    def __init__(self,color,width,cap,join,dash):
        """ Creates a new `LineState` object.

            All input arguments that are containers are copied
            by the constructor.  This prevents two `LineState` objects
            from ever sharing and modifying the other's data.
        """
        self.line_color = array(color,copy=1)
        self.line_width     = width
        self.line_cap       = cap
        self.line_join      = join
        if not dash:
            # always set line_dash to be a tuple
            self.line_dash  = NO_DASH
        else:
            self.line_dash  = (dash[0],array(dash[1],copy=1))

    def copy(self):
        """ Makes a copy of the current line state. Could just use
            deepcopy...
        """
        return LineState(self.line_color, self.line_width,
                          self.line_cap  , self.line_join,
                          self.line_dash)

    def is_dashed(self):
        # if line_dash only has one entry, it is a solid line.
        return is_dashed(self.line_dash)

class GraphicsState(LineState):
    """ Holds information used by a graphics context when drawing.

        I'm not sure if these should be a separate class, a dictionary,
        or part of the GraphicsContext object.  Making them a dictionary
        or object simplifies save_state and restore_state a little bit.

        Also, this is a pretty good candidate for using slots.  I'm not
        going to use them right now, but, if we standardize on 2.2, slots might
        speed things up some.

        Fields
        ------

        ctm
            context transform matrix

        These are inherited from LineState:

        line_color
            RGBA array(4) of values 0.0 to 1.0
        line_width
            width of drawn lines
        line_join
            style of how lines are joined.  The choices
            are: JOIN_ROUND, JOIN_BEVEL, JOIN_MITER
        line_cap
            style of the end cap on lines.  The choices
            are: CAP_ROUND, CAP_SQUARE, CAP_BUTT
        line_dash
            (phase,pattern) dash pattern for lines.
            phase is a single value specifying how many
            units into the pattern to start.  dash is
            a 1-D array of floats that alternate between
            specifying the number of units on and off
            in the pattern.  When the end of the array
            is reached, the pattern repeats.
        fill_color
            RGBA array(4) of values 0.0 to 1.0
        alpha
            transparency value of drawn objects
        font
            either a special device independent font
            object (what does anygui use?) or a
            device dependent font object.
        text_matrix
            coordinate transformation matrix for text
        clipping_path
            defines the path of the clipping region.
            For now, this can only be a rectangle.
        current_point
            location where next object is drawn.
        should_antialias
            whether anti-aliasing should be used when
            drawing lines and fonts
        miter_limit
            specifies when and when not to miter line joins.
        flatness
            not sure
        character_spacing
            spacing between drawing text characters
        text_drawing_mode
            style for drawing text: outline, fill, etc.

        Not yet supported:

        rendering_intent
            deals with colors and color correction in
            a sophisticated way.
    """
    def __init__(self):

        #---------------------------------------------------------------------
        # Line state default values.
        #---------------------------------------------------------------------
        line_color     = array( ( 0.0, 0.0, 0.0, 1.0 ) )
        line_width     = 1
        line_cap       = CAP_ROUND
        line_join      = JOIN_MITER
        line_dash      = ( 0, array( [ 0 ] ) ) # This will draw a solid line
        LineState.__init__( self, line_color, line_width, line_cap,
                                  line_join, line_dash )

        #---------------------------------------------------------------------
        # All other default values.
        #---------------------------------------------------------------------
        self.ctm              = affine.affine_identity()
        self.fill_color       = array( ( 0.0, 0.0, 0.0, 1.0 ) )
        self.alpha            = 1.0
#        self.font             = freetype.FontInfo(
#                                   freetype.default_font_info.default_font )
        self.font = None
        self.text_matrix      = affine.affine_identity()
        self.clipping_path    = None # Not sure what the default should be?
        # Technically uninitialized in the PDF spec, but 0,0 seems fine to me:
        self.current_point     = array( ( 0, 0 ), float64 )

        self.antialias         = 1
        # What should this default to?
        self.miter_limit       = 1.0
        # Not so sure about this one either.
        self.flatness          = None
        # I think this is the correct default.
        self.character_spacing = 0.0
        # Should it be outline also?
        self.text_drawing_mode = TEXT_FILL
        self.alpha             = 1.0

    def copy(self):
        return copy.deepcopy(self)

class GraphicsContextBase(object):
    """

        Fields
        ------

        state
            Current state of graphics context.
        state_stack
            Stack used to save graphics states
        path
            The drawing path.
        active_subpath
            The active drawing subpath

        I *think* this class needs to be sub-classed by every device type
        that handles graphics.  This is so that set_line_width() and similar
        functions can do things like setting up a new pen in wxPython, etc.
        This stuff could also be stored in the GraphicsState, and there
        are probably performance benefits for doing so.  Maybe graphics
        state is the device dependent object??  Time will tell.

        path and active_subpath will probably need to be optimized somehow.
    """

    def __init__(self, *args, **kwargs):
        super(GraphicsContextBase, self).__init__()
        self.state = GraphicsState()

        # The line state has multiple properties that are tracked by a class
        self.last_drawn_line_state = LineState(None,None,None,None,None)

        # The fill state is simply a color.
        self.last_drawn_fill_state = None
        self.last_font_state = None

        # Used by save/restore state.
        self.state_stack = []

        # Variables for used in drawing paths.
        # The path_transform_indices holds a list of indices pointing into
        # active_subpath that affect the ctm.  It is necessary to preserve
        # these across begin_path calls.
        self.active_subpath = []
        self.path_transform_indices = []
        self.path = [self.active_subpath]

        # Used as memory cache for transforming points
        #self.transform_points_cache = array((2,1))

        # Whether the particular underlying graphics context considers the
        # "origin" of a pixel to be the center of the pixel or the lower-left
        # corner.  Most vector-based drawing systems consider the origin to
        # be at the corner, whereas most raster systems place the origin at
        # the center.
        #
        # This is ultimately used to determine whether certain methods should
        # automatically tack on a (0.5, 0.5) offset.
        self.corner_pixel_origin = True

        #--------------------------------------------------------------------
        # We're currently maintaining a couple of copies of the ctm around.
        # The state.ctm is used mainly for user querying, etc.  We also have
        # something called the device_ctm which is actually used in the
        # drawing of objects.  In some implementation (OpenGL), the
        # device_ctm is actually maintained in hardware.
        #--------------------------------------------------------------------
        self.device_prepare_device_ctm()

    #------------------------------------------------------------------------
    # Coordinate Transform Matrix Manipulation
    #
    # Note:  I'm not sure we really need to keep the state.ctm around now
    #        that we're keeping the device_ctm around, but I'm reluctant to
    #        unify the two yet.  I think it can (and probably should) be done
    #        though.
    #------------------------------------------------------------------------

    def scale_ctm(self, sx, sy):
        """ Sets the coordinate system scale to the given values, (sx,sy).

            Parameters
            ----------
            sx : float
                The new scale factor for the x axis
            sy : float
                The new scale factor for the y axis
        """
        self.state.ctm = affine.scale(self.state.ctm,sx,sy)
        self.active_subpath.append( (SCALE_CTM, (sx,sy)) )
        self.path_transform_indices.append(len(self.active_subpath)-1)

    def translate_ctm(self, tx, ty):
        """ Translates the coordinate system by the value given by (tx,ty)

            Parameters
            ----------
            tx : float
                The distance to move in the x direction
            ty : float
                The distance to move in the y direction
        """
        self.state.ctm = affine.translate(self.state.ctm,tx,ty)
        self.active_subpath.append( (TRANSLATE_CTM, (tx,ty)) )
        self.path_transform_indices.append(len(self.active_subpath)-1)

    def rotate_ctm(self, angle):
        """ Rotates the coordinate space for drawing by the given angle.

            Parameters
            ----------
            angle : float
                the angle, in radians, to rotate the coordinate system
        """
        self.state.ctm = affine.rotate(self.state.ctm,angle)
        self.active_subpath.append( (ROTATE_CTM, (angle,)) )
        self.path_transform_indices.append(len(self.active_subpath)-1)

    def concat_ctm(self, transform):
        """ Concatenates the transform to current coordinate transform matrix.

            Parameters
            ----------
            transform : affine_matrix
                the transform matrix to concatenate with
                the current coordinate matrix.
        """
        self.state.ctm = affine.concat(self.state.ctm,transform)
        self.active_subpath.append( (CONCAT_CTM, (transform,)) )
        self.path_transform_indices.append(len(self.active_subpath)-1)

    def get_ctm(self):
        """ Returns the current coordinate transform matrix.
        """
        return self.state.ctm.copy()

    #----------------------------------------------------------------
    # Save/Restore graphics state.
    #----------------------------------------------------------------

    def save_state(self):
        """ Saves the current graphic's context state.

            Always pair this with a `restore_state()`.
        """
        self.state_stack.append(self.state)
        self.state = self.state.copy()

    def restore_state(self):
        """ Restores the previous graphics state.
        """
        self.state = self.state_stack.pop(-1)
        self.active_subpath.append( (LOAD_CTM, (self.state.ctm,)) )
        self.path_transform_indices.append(len(self.active_subpath)-1)

    #----------------------------------------------------------------
    # context manager interface
    #----------------------------------------------------------------

    def __enter__(self):
        self.save_state()

    def __exit__(self, type, value, traceback):
        self.restore_state()

    #----------------------------------------------------------------
    # Manipulate graphics state attributes.
    #----------------------------------------------------------------

    def set_antialias(self,value):
        """ Sets/Unsets anti-aliasing for bitmap graphics context.

            Ignored on most platforms.
        """
        self.state.antialias = value

    def set_line_width(self,width):
        """ Sets the line width for drawing

            Parameters
            ----------
            width : float
                The new width for lines in user space units.
        """
        self.state.line_width = width

    def set_line_join(self,style):
        """ Sets the style for joining lines in a drawing.

            Parameters
            ----------
            style : join_style
                The line joining style.  The available
                styles are JOIN_ROUND, JOIN_BEVEL, JOIN_MITER.
        """
        if style not in (JOIN_ROUND,JOIN_BEVEL,JOIN_MITER):
            msg = "Invalid line join style.  See documentation for valid styles"
            raise ValueError, msg
        self.state.line_join = style

    def set_miter_limit(self,limit):
        """ Specifies limits on line lengths for mitering line joins.

            If line_join is set to miter joins, the limit specifies which
            line joins should actually be mitered.  If lines are not mitered,
            they are joined with a bevel.  The line width is divided by
            the length of the miter.  If the result is greater than the
            limit, the bevel style is used.

            This is not implemented on most platforms.

            Parameters
            ----------
            limit : float
                limit for mitering joins. defaults to 1.0.
                (XXX is this the correct default?)
        """
        self.state.miter_limit = limit

    def set_line_cap(self,style):
        """ Specifies the style of endings to put on line ends.

            Parameters
            ----------
            style : cap_style
                The line cap style to use. Available styles
                are CAP_ROUND, CAP_BUTT, CAP_SQUARE.
        """
        if style not in (CAP_ROUND,CAP_BUTT,CAP_SQUARE):
            msg = "Invalid line cap style.  See documentation for valid styles"
            raise ValueError, msg
        self.state.line_cap = style

    def set_line_dash(self,pattern,phase=0):
        """ Sets the line dash pattern and phase for line painting.

            Parameters
            ----------
            pattern : float array
                An array of floating point values
                specifing the lengths of on/off painting
                pattern for lines.
            phase : float
                Specifies how many units into dash pattern
                to start.  phase defaults to 0.
        """
        if not alltrue(pattern):
            self.state.line_dash = NO_DASH
            return
        pattern = asarray(pattern)
        if len(pattern) < 2:
            raise ValueError, "dash pattern should have at least two entries."
        # not sure if this check is really needed.
        if phase < 0:
            raise ValueError, "dash phase should be a positive value."
        self.state.line_dash = (phase,pattern)

    def set_flatness(self,flatness):
        """ Not implemented

            It is device dependent and therefore not recommended by
            the PDF documentation.

            flatness determines how accurately lines are rendered.  Setting it
            to values less than one will result in more accurate drawings, but
            they take longer.  It defaults to None
        """
        self.state.flatness = flatness

    #----------------------------------------------------------------
    # Sending drawing data to a device
    #----------------------------------------------------------------

    def flush(self):
        """ Sends all drawing data to the destination device.

            Currently this is a NOP for wxPython.
        """
        pass

    def synchronize(self):
        """ Prepares drawing data to be updated on a destination device.

            Currently this is a NOP for all implementations.
        """
        pass

    #----------------------------------------------------------------
    # Page Definitions
    #----------------------------------------------------------------

    def begin_page(self):
        """ Creates a new page within the graphics context.

            Currently this is a NOP for all implementations.  The PDF
            backend should probably implement it, but the ReportLab
            Canvas uses the showPage() method to handle both
            begin_page and end_page issues.
        """
        pass

    def end_page(self):
        """ Ends drawing in the current page of the graphics context.

            Currently this is a NOP for all implementations.  The PDF
            backend should probably implement it, but the ReportLab
            Canvas uses the showPage() method to handle both
            begin_page and end_page issues.
        """
        pass

    #----------------------------------------------------------------
    # Building paths (contours that are drawn)
    #
    # + Currently, nothing is drawn as the path is built.  Instead, the
    #   instructions are stored and later drawn.  Should this be changed?
    #   We will likely draw to a buffer instead of directly to the canvas
    #   anyway.
    #
    #   Hmmm. No.  We have to keep the path around for storing as a
    #   clipping region and things like that.
    #
    # + I think we should keep the current_path_point hanging around.
    #
    #----------------------------------------------------------------

    def begin_path(self):
        """ Clears the current drawing path and begin a new one.
        """
        # Need to check here if the current subpath contains matrix
        # transforms.  If  it does, pull these out, and stick them
        # in the new subpath.
        if self.path_transform_indices:
            #print 'begin'
            #print self.path_transform_indices
            #print len(self.active_subpath)
            #tf = take(array(self.active_subpath,object),
            #          self.path_transform_indices)
            tf = array(self.active_subpath, object)[self.path_transform_indices, :]
            self.path_transform_indices = range(len(tf))
            self.active_subpath = list(tf)
        else:
            self.active_subpath = []
        self.path = [self.active_subpath]

    def move_to(self,x,y):
        """ Starts a new drawing subpath and place the current point at (x,y).

            Notes:
                Not sure how to treat state.current_point.  Should it be the
                value of the point before or after the matrix transformation?
                It looks like before in the PDF specs.
        """
        self._new_subpath()

        pt = array((x,y),float64)
        self.state.current_point = pt
        #pt = affine.transform_point(self.get_ctm(),orig)
        self.active_subpath.append( (POINT, pt) )

    def line_to(self,x,y):
        """ Adds a line from the current point to the given point (x,y).

            The current point is moved to (x,y).

            What should happen if move_to hasn't been called? Should it always
            begin at 0,0 or raise an error?

            Notes:
                See note in move_to about the current_point.
        """
        pt = array((x,y),float64)
        self.state.current_point = pt
        #pt = affine.transform_point(self.get_ctm(),orig)
        self.active_subpath.append( (LINE, pt ) )

    def lines(self,points):
        """ Adds a series of lines as a new subpath.

            Parameters
            ----------

            points
                an Nx2 array of x,y pairs

            The current_point is moved to the last point in 'points'
        """
        self._new_subpath()
        pts = points
        #pts = affine.transform_points(self.get_ctm(),points)
        self.active_subpath.append( (LINES,pts) )
        self.state.current_point = points[-1]

    def line_set(self, starts, ends):
        """ Adds a set of disjoint lines as a new subpath.

            Parameters
            ----------
            starts
                an Nx2 array of x,y pairs
            ends
                an Nx2 array of x,y pairs

            Starts and ends should have the same length.
            The current point is moved to the last point in 'ends'.
        """
        self._new_subpath()
        for i in xrange(min(len(starts), len(ends))):
            self.active_subpath.append( (POINT, starts[i]) )
            self.active_subpath.append( (LINE, ends[i]) )
        self.state.current_point = ends[i]

    def rect(self,x,y,sx,sy):
        """ Adds a rectangle as a new subpath.
        """
        pts = array(((x   ,y   ),
                     (x   ,y+sy),
                     (x+sx,y+sy),
                     (x+sx,y   ),))
        self.lines(pts)
        self.close_path('rect')

    def draw_rect(self, rect, mode):
        self.rect(*rect)
        self.draw_path(mode=mode)

    def rects(self,rects):
        """ Adds multiple rectangles as separate subpaths to the path.

            Not very efficient -- calls rect multiple times.
        """
        for x,y,sx,sy in rects:
            self.rect(x,y,sx,sy)

    def close_path(self,tag=None):
        """ Closes the path of the current subpath.

            Currently starts a new subpath -- is this what we want?
        """
        self.active_subpath.append((CLOSE,(tag,)))
        self._new_subpath()

    def curve_to(self, x_ctrl1, y_ctrl1, x_ctrl2, y_ctrl2, x_to, y_to):
        """ Draw a cubic bezier curve from the current point.

        Parameters
        ----------
        x_ctrl1 : float
            X-value of the first control point.
        y_ctrl1 : float
            Y-value of the first control point.
        x_ctrl2 : float
            X-value of the second control point.
        y_ctrl2 : float
            Y-value of the second control point.
        x_to : float
            X-value of the ending point of the curve.
        y_to : float
            Y-value of the ending point of the curve.
        """
        # XXX: figure out a reasonable number of points from the current scale
        # and arc length. Since the arc length is expensive to calculate, the
        # sum of the lengths of the line segments from (xy0, xy_ctrl1),
        # (xy_ctrl1, xy_ctrl2), and (xy_ctrl2, xy_to) would be a reasonable
        # approximation.
        n = 100
        t = np.arange(1, n+1) / float(n)
        t2 = t*t
        t3 = t2*t
        u = 1 - t
        u2 = u*u
        u3 = u2*u
        x0, y0 = self.state.current_point
        pts = np.column_stack([
            x0*u3 + 3*(x_ctrl1*t*u2 + x_ctrl2*t2*u) + x_to*t3,
            y0*u3 + 3*(y_ctrl1*t*u2 + y_ctrl2*t2*u) + y_to*t3,
        ])
        self.active_subpath.append( (LINES,pts) )
        self.state.current_point = pts[-1]

    def quad_curve_to(self, x_ctrl, y_ctrl, x_to, y_to):
        """ Draw a quadratic bezier curve from the current point.

        Parameters
        ----------
        x_ctrl : float
            X-value of the control point
        y_ctrl : float
            Y-value of the control point.
        x_to : float
            X-value of the ending point of the curve
        y_to : float
            Y-value of the ending point of the curve.
        """
        # A quadratic Bezier curve is just a special case of the cubic. Reuse
        # its implementation in case it has been implemented for the specific
        # backend.
        x0, y0 = self.state.current_point
        xc1 = (x0 + x_ctrl + x_ctrl) / 3.0
        yc1 = (y0 + y_ctrl + y_ctrl) / 3.0
        xc2 = (x_to + x_ctrl + x_ctrl) / 3.0
        yc2 = (y_to + y_ctrl + y_ctrl) / 3.0
        self.curve_to(xc1, yc1, xc2, yc2, x_to, y_to)

    def arc(self, x, y, radius, start_angle, end_angle, cw=False):
        """ Draw a circular arc.

        If there is a current path and the current point is not the initial
        point of the arc, a line will be drawn to the start of the arc. If there
        is no current path, then no line will be drawn.

        Parameters
        ----------
        x : float
            X-value of the center of the arc.
        y : float
            Y-value of the center of the arc.
        radius : float
            The radius of the arc.
        start_angle : float
            The angle, in radians, that the starting point makes with respect
            to the positive X-axis from the center point.
        end_angle : float
            The angles, in radians, that the final point makes with
            respect to the positive X-axis from the center point.
        cw : bool, optional
            Whether the arc should be drawn clockwise or not.
        """
        # XXX: pick the number of line segments based on the current scale and
        # the radius.
        n = 100
        if end_angle < start_angle and not cw:
            end_angle += 2*pi
        elif start_angle < end_angle and cw:
            start_angle += 2*pi
        theta = np.linspace(start_angle, end_angle, n)
        pts = radius * np.column_stack([np.cos(theta), np.sin(theta)])
        pts += np.array([x, y])
        self.active_subpath.append( (LINES,pts) )
        self.state.current_point = pts[-1]

    def arc_to(self, x1, y1, x2, y2, radius):
        """
        """
        raise NotImplementedError, "arc_to is not implemented"

    def _new_subpath(self):
        """ Starts a new drawing subpath.

            Only creates a new subpath if the current one contains objects.
        """
        if self.active_subpath:
            self.active_subpath = []
            self.path_transform_indices = []
            self.path.append(self.active_subpath)

    #----------------------------------------------------------------
    # Getting infomration on paths
    #----------------------------------------------------------------

    def is_path_empty(self):
        """ Tests to see whether the current drawing path is empty
        """
        # If the first subpath is empty, then the path is empty
        res = 0
        if not self.path[0]:
            res = 1
        else:
            res = 1
            for sub in self.path:
                if not is_point(sub[-1]):
                    res = 0
                    break
        return res


    def get_path_current_point(self):
        """ Returns the current point from the graphics context.

            Note:
                Currently the current_point is only affected by move_to,
                line_to, and lines.  It should also be affected by text
                operations.  I'm not sure how rect and rects and friends
                should affect it -- will find out on Mac.
        """
        pass

    def get_path_bounding_box(self):
        """
        """
        pass

    def from_agg_affine(self, aff):
        """Convert an agg.AffineTransform to a numpy matrix
        representing the affine transform usable by kiva.affine
        and other non-agg parts of kiva"""
        return array([[aff[0], aff[1], 0],
                      [aff[2], aff[3], 0],
                      [aff[4], aff[5], 1]], float64)

    def add_path(self, path):
        """Draw a compiled path into this gc.  Note: if the CTM is
        changed and not restored to the identity in the compiled path,
        the CTM change will continue in this GC."""
        # Local import to avoid a dependency if we can avoid it.
        from kiva import agg

        multi_state = 0 #For multi-element path commands we keep the previous
        x_ctrl1 = 0     #information in these variables.
        y_ctrl1 = 0
        x_ctrl2 = 0
        y_ctrl2 = 0
        for x, y, cmd, flag in path._vertices():
            if cmd == agg.path_cmd_line_to:
                self.line_to(x,y)
            elif cmd == agg.path_cmd_move_to:
                self.move_to(x, y)
            elif cmd == agg.path_cmd_stop:
                self.concat_ctm(path.get_kiva_ctm())
            elif cmd == agg.path_cmd_end_poly:
                self.close_path()
            elif cmd == agg.path_cmd_curve3:
                if multi_state == 0:
                    x_ctrl1 = x
                    y_ctrl1 = y
                    multi_state = 1
                else:
                    self.quad_curve_to(x_ctrl1, y_ctrl1, x, y)
                    multi_state = 0
            elif cmd == agg.path_cmd_curve4:
                if multi_state == 0:
                    x_ctrl1 = x
                    y_ctrl1 = y
                    multi_state = 1
                elif multi_state == 1:
                    x_ctrl2 = x
                    y_ctrl2 = y
                    multi_state = 2
                elif multi_state == 2:
                    self.curve_to(x_ctrl1, y_ctrl1, x_ctrl2, y_ctrl2, x, y)



    #----------------------------------------------------------------
    # Clipping path manipulation
    #----------------------------------------------------------------

    def clip(self):
        """
        """
        pass

    def even_odd_clip(self):
        """
        """
        pass


    def clip_to_rect(self,x,y,width,height):
        """
            Sets the clipping path to the intersection of the current clipping
            path with the area defined by the specified rectangle
        """
        if not self.state.clipping_path:
            self.state.clipping_path = ( x, y, width, height )
            self.device_set_clipping_path( x, y, width, height )
        else:
            # Find the intersection of the clipping regions:
            xmin1, ymin1, width1, height1 = self.state.clipping_path
            xclip_min = max( xmin1, x )
            xclip_max = min( xmin1 + width1, x + width )
            yclip_min = max( ymin1, y )
            yclip_max = min( ymin1 + height1, y + height )
            height_clip = max( 0, yclip_max - yclip_min )
            width_clip  = max( 0, xclip_max - xclip_min )
            self.state.clipping_path = ( xclip_min,  yclip_min,
                                         width_clip, height_clip )
            self.device_set_clipping_path( xclip_min,  yclip_min,
                                           width_clip, height_clip )

    def clip_to_rects(self):
        """
        """
        pass

    def clear_clip_path(self):
        self.state.clipping_path=None
        self.device_destroy_clipping_path()

    #----------------------------------------------------------------
    # Color space manipulation
    #
    # I'm not sure we'll mess with these at all.  They seem to
    # be for setting the color system.  Hard coding to RGB or
    # RGBA for now sounds like a reasonable solution.
    #----------------------------------------------------------------

    #def set_fill_color_space(self):
    #    """
    #    """
    #    pass

    #def set_stroke_color_space(self):
    #    """
    #    """
    #    pass

    #def set_rendering_intent(self):
    #    """
    #    """
    #    pass

    #----------------------------------------------------------------
    # Color manipulation
    #----------------------------------------------------------------

    def set_fill_color(self,color):
        """
            set_fill_color takes a sequences of rgb or rgba values
            between 0.0 and 1.0
        """
        if len(color) == 3:
            self.state.fill_color[:3]= color
            self.state.fill_color[3]= 1.0
        else:
            self.state.fill_color[:]= color


    def set_stroke_color(self,color):
        """
            set_stroke_color takes a sequences of rgb or rgba values
            between 0.0 and 1.0
        """
        if len(color) == 3:
            self.state.line_color[:3]= color
            self.state.line_color[3]= 1.0
        else:
            self.state.line_color[:]= color

    def set_alpha(self,alpha):
        """
        """
        self.state.alpha = alpha

    #def set_gray_fill_color(self):
    #    """
    #    """
    #    pass

    #def set_gray_stroke_color(self):
    #    """
    #    """
    #    pass

    #def set_rgb_fill_color(self):
    #    """
    #    """
    #    pass

    #def set_rgb_stroke_color(self):
    #    """
    #    """
    #    pass

    #def cmyk_fill_color(self):
    #    """
    #    """
    #    pass

    #def cmyk_stroke_color(self):
    #    """
    #    """
    #    pass

    #----------------------------------------------------------------
    # Drawing Images
    #----------------------------------------------------------------

    def draw_image(self,img,rect=None):
        """
        """
        self.device_draw_image(img, rect)

    #----------------------------------------------------------------
    # Drawing PDF documents
    #----------------------------------------------------------------

    #def draw_pdf_document(self):
    #    """
    #    """
    #    pass

    #-------------------------------------------------------------------------
    # Drawing Text
    #
    # Font handling needs more attention.
    #
    #-------------------------------------------------------------------------

    def select_font(self,face_name,size=12,style="regular",encoding=None):
        """ Selects a new font for drawing text.

            Parameters
            ----------

            face_name
                The name of a font. E.g.: "Times New Roman"
                !! Need to specify a way to check for all the types
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
        # !! should check if name and encoding are valid.
#        self.state.font = freetype.FontInfo(face_name,size,style,encoding)
        self.state.font = None

    def set_font(self,font):
        """ Set the font for the current graphics context.
        """
        self.state.font = font.copy()

    def set_font_size(self,size):
        """ Sets the size of the font.

            The size is specified in user space coordinates.

            Note:
                I don't think the units of this are really "user space
                coordinates" on most platforms.  I haven't looked into
                the text drawing that much, so this stuff needs more
                attention.
        """
        return

    def set_character_spacing(self,spacing):
        """ Sets the amount of additional spacing between text characters.

            Parameters
            ----------

            spacing : float
                units of space extra space to add between
                text coordinates.  It is specified in text coordinate
                system.

            Notes
            -----
            1.  I'm assuming this is horizontal spacing?
            2.  Not implemented in wxPython.
        """
        self.state.character_spacing = spacing


    def set_text_drawing_mode(self, mode):
        """ Specifies whether text is drawn filled or outlined or both.

            Parameters
            ----------

            mode
                determines how text is drawn to the screen.  If
                a CLIP flag is set, the font outline is added to the
                clipping path. Possible values:

                    TEXT_FILL
                        fill the text
                    TEXT_STROKE
                        paint the outline
                    TEXT_FILL_STROKE
                        fill and outline
                    TEXT_INVISIBLE
                        paint it invisibly ??
                    TEXT_FILL_CLIP
                        fill and add outline clipping path
                    TEXT_STROKE_CLIP
                        outline and add outline to clipping path
                    TEXT_FILL_STROKE_CLIP
                        fill, outline, and add to clipping path
                    TEXT_CLIP
                        add text outline to clipping path

            Note:
                wxPython currently ignores all but the INVISIBLE flag.
        """
        if mode not in (TEXT_FILL, TEXT_STROKE, TEXT_FILL_STROKE,
                        TEXT_INVISIBLE, TEXT_FILL_CLIP, TEXT_STROKE_CLIP,
                        TEXT_FILL_STROKE_CLIP, TEXT_CLIP, TEXT_OUTLINE):
            msg = "Invalid text drawing mode.  See documentation for valid modes"
            raise ValueError, msg
        self.state.text_drawing_mode = mode

    def set_text_position(self,x,y):
        """
        """
        a,b,c,d,tx,ty = affine.affine_params(self.state.text_matrix)
        tx, ty = x,y
        self.state.text_matrix = affine.affine_from_values(a,b,c,d,tx,ty)
        # No longer uses knowledge that matrix has 3x3 representation
        #self.state.text_matrix[2,:2] = (x,y)

    def get_text_position(self):
        """
        """
        a,b,c,d,tx,ty = affine.affine_params(self.state.text_matrix)
        return tx,ty
        # No longer uses knowledge that matrix has 3x3 representation
        #return self.state.text_matrix[2,:2]

    def set_text_matrix(self,ttm):
        """
        """
        self.state.text_matrix = ttm.copy()

    def get_text_matrix(self):
        """
        """
        return self.state.text_matrix.copy()

    def show_text(self,text):
        """ Draws text on the device at the current text position.

            This calls the device dependent device_show_text() method to
            do all the heavy lifting.

            It is not clear yet how this should affect the current point.
        """
        self.device_show_text(text)

    #------------------------------------------------------------------------
    # kiva defaults to drawing text using the freetype rendering engine.
    #
    # If you would like to use a systems native text rendering engine,
    # override this method in the class concrete derived from this one.
    #------------------------------------------------------------------------
    def device_show_text(self,text):
        """ Draws text on the device at the current text position.

            This relies on the FreeType engine to render the text to an array
            and then calls the device dependent device_show_text() to display
            the rendered image to the screen.

            !! antiliasing is turned off until we get alpha blending
            !! of images figured out.
        """

        # This is not currently implemented in a device-independent way.
        return

        ##---------------------------------------------------------------------
        ## The fill_color is used to specify text color in wxPython.
        ## If it is transparent, we don't do any painting.
        ##---------------------------------------------------------------------
        #if is_fully_transparent( self.state.fill_color ):
        #   return
        #
        ##---------------------------------------------------------------------
        ## Set the text transformation matrix
        ##
        ## This requires the concatenation of the text and coordinate
        ## transform matrices
        ##---------------------------------------------------------------------
        #ttm = self.get_text_matrix()
        #ctm = self.get_ctm()  # not device_ctm!!
        #m   = affine.concat( ctm, ttm )
        #a, b, c, d, tx, ty = affine.affine_params( m )
        #ft_engine.transform( ( a, b, c, d ) )
        #
        ## Select the correct font into the freetype engine:
        #f = self.state.font
        #ft_engine.select_font( f.name, f.size, f.style, f.encoding )
        #ft_engine.select_font( 'Arial', 10 )   ### TEMPORARY ###
        #
        ## Set antialiasing flag for freetype engine:
        #ft_engine.antialias( self.state.antialias )
        #
        ## Render the text:
        ##
        ## The returned object is a freetype.Glyphs object that contains an
        ## array with the gray scale image, the bbox and some other info.
        #rendered_glyphs = ft_engine.render( text )
        #
        ## Render the glyphs in a device specific manner:
        #self.device_draw_glyphs( rendered_glyphs, tx, ty )
        #
        ## Advance the current text position by the width of the glyph string:
        #ttm = self.get_text_matrix()
        #a, b, c, d, tx, ty = affine.affine_params( ttm )
        #tx += rendered_glyphs.width
        #self.state.text_matrix = affine.affine_from_values( a, b, c, d, tx, ty )

    def show_glyphs(self):
        """
        """
        pass

    def show_text_at_point(self, text, x, y):
        """
        """
        pass

    def show_glyphs_at_point(self):
        """
        """
        pass

    #----------------------------------------------------------------
    # Painting paths (drawing and filling contours)
    #----------------------------------------------------------------

    def stroke_path(self):
        self.draw_path(mode=STROKE)

    def fill_path(self):
        self.draw_path(mode=FILL)

    def eof_fill_path(self):
        self.draw_path(mode=EOF_FILL)

    def draw_path(self, mode=FILL_STROKE):
        """ Walks through all the drawing subpaths and draw each element.

            Each subpath is drawn separately.

            Parameters
            ----------
            mode
                Specifies how the subpaths are drawn.  The default is
                FILL_STROKE.  The following are valid values.

                    FILL
                        Paint the path using the nonzero winding rule
                        to determine the regions for painting.
                    EOF_FILL
                        Paint the path using the even-odd fill rule.
                    STROKE
                        Draw the outline of the path with the
                        current width, end caps, etc settings.
                    FILL_STROKE
                        First fill the path using the nonzero
                        winding rule, then stroke the path.
                    EOF_FILL_STROKE
                        First fill the path using the even-odd
                        fill method, then stroke the path.
        """
        #---------------------------------------------------------------------
        # FILL AND STROKE settings are handled by setting the alpha value of
        # the line and fill colors to zero (transparent) if stroke or fill
        # is not needed.
        #---------------------------------------------------------------------

        old_line_alpha = self.state.line_color[3]
        old_fill_alpha = self.state.fill_color[3]
        if mode not in [STROKE, FILL_STROKE, EOF_FILL_STROKE]:
            self.state.line_color[3] = 0.0
        if mode not in [FILL, EOF_FILL, FILL_STROKE, EOF_FILL_STROKE]:
            self.state.fill_color[3] = 0.0

        #print 'in:',self.device_ctm
        self.device_update_line_state()
        self.device_update_fill_state()

        for subpath in self.path:
            # reset the current point for drawing.
            #self.current_point = array((0.,0.))
            self.clear_subpath_points()
            for func,args in subpath:
                if func == POINT:
                    self.draw_subpath(mode)
                    self.add_point_to_subpath(args)
                    self.first_point = args
                elif func == LINE:
                    self.add_point_to_subpath(args)
                elif func == LINES:
                    self.draw_subpath(mode)
                    # add all points in list to subpath.
                    self.add_point_to_subpath(args)
                    self.first_point = args[0]
                elif func == CLOSE:
                    self.add_point_to_subpath(self.first_point)
                    self.draw_subpath(mode)
                elif func == RECT:
                    self.draw_subpath(mode)
                    self.device_draw_rect(args[0],args[1],args[2],args[3],
                                          mode)
                elif func in [SCALE_CTM,ROTATE_CTM,TRANSLATE_CTM,
                              CONCAT_CTM,LOAD_CTM]:
                    self.device_transform_device_ctm(func,args)
                else:
                    print 'oops:', func
            # finally, draw any remaining paths.
            self.draw_subpath(mode)

        #---------------------------------------------------------------------
        # reset the alpha values for line and fill values.
        #---------------------------------------------------------------------
        self.state.line_color[3] = old_line_alpha
        self.state.fill_color[3] = old_fill_alpha

        #---------------------------------------------------------------------
        # drawing methods always consume the path on Mac OS X.  We'll follow
        # this convention to make implementation there easier.
        #---------------------------------------------------------------------
        self.begin_path()

    def device_prepare_device_ctm(self):
        self.device_ctm = affine.affine_identity()

    def device_transform_device_ctm(self,func,args):
        """ Default implementation for handling scaling matrices.

            Many implementations will just use this function.  Others, like
            OpenGL, can benefit from overriding the method and using
            hardware acceleration.
        """
        if func == SCALE_CTM:
            #print  'scale:', args
            self.device_ctm = affine.scale(self.device_ctm,args[0],args[1])
        elif func == ROTATE_CTM:
            #print 'rotate:', args
            self.device_ctm = affine.rotate(self.device_ctm,args[0])
        elif func == TRANSLATE_CTM:
            #print 'translate:', args
            self.device_ctm = affine.translate(self.device_ctm,args[0],args[1])
        elif func == CONCAT_CTM:
            #print  'concat'
            self.device_ctm = affine.concat(self.device_ctm,args[0])
        elif func == LOAD_CTM:
            #print 'load'
            self.device_ctm = args[0].copy()

    def device_draw_rect(self,x,y,sx,sy,mode):
        """ Default implementation of drawing  a rect.
        """
        self._new_subpath()
        # When rectangles are rotated, they have to be drawn as a polygon
        # on most devices.  We'll need to specialize this on API's that
        # can handle rotated rects such as Quartz and OpenGL(?).
        # All transformations are done in the call to lines().
        pts = array(((x   ,y   ),
                     (x   ,y+sy),
                     (x+sx,y+sy),
                     (x+sx,y   ),
                     (x   ,y   )))
        self.add_point_to_subpath(pts)
        self.draw_subpath(mode)

    def stroke_rect(self):
        """
        """
        pass

    def stroke_rect_with_width(self):
        """
        """
        pass

    def fill_rect(self):
        """
        """
        pass

    def fill_rects(self):
        """
        """
        pass

    def clear_rect(self):
        """
        """
        pass

    #----------------------------------------------------------------
    # Subpath point management and drawing routines.
    #----------------------------------------------------------------

    def add_point_to_subpath(self,pt):
        self.draw_points.append(pt)

    def clear_subpath_points(self):
        self.draw_points = []

    def get_subpath_points(self,debug=0):
        """ Gets the points that are in the current path.

            The first entry in the draw_points list may actually
            be an array.  If this is true, the other points are
            converted to an array and concatenated with the first
        """
        if self.draw_points and len(shape(self.draw_points[0])) > 1:
            first_points = self.draw_points[0]
            other_points = asarray(self.draw_points[1:])
            if len(other_points):
                pts = concatenate((first_points,other_points),0)
            else:
                pts = first_points
        else:
            pts = asarray(self.draw_points)
        return pts

    def draw_subpath(self,mode):
        """ Fills and strokes the point path.

            After the path is drawn, the subpath point list is
            cleared and ready for the next subpath.

            Parameters
            ----------

            mode
                Specifies how the subpaths are drawn.  The default is
                FILL_STROKE.  The following are valid values.

                    FILL
                        Paint the path using the nonzero winding rule
                        to determine the regions for painting.
                    EOF_FILL
                        Paint the path using the even-odd fill rule.
                    STROKE
                        Draw the outline of the path with the
                        current width, end caps, etc settings.
                    FILL_STROKE
                        First fill the path using the nonzero
                        winding rule, then stroke the path.
                    EOF_FILL_STROKE
                        First fill the path using the even-odd
                        fill method, then stroke the path.

            Note:
                If path is closed, it is about 50% faster to call
                DrawPolygon with the correct pen set than it is to
                call DrawPolygon and DrawLines separately in wxPython. But,
                because paths can be left open, Polygon can't be
                called in the general case because it automatically
                closes the path.  We might want a separate check in here
                and allow devices to specify a faster version if the path is
                closed.
        """
        pts = self.get_subpath_points()
        if len(pts) > 1:
            self.device_fill_points(pts,mode)
            self.device_stroke_points(pts,mode)
        self.clear_subpath_points()


    def get_text_extent(self,textstring):
        """
            Calls device specific text extent method.
        """
        return self.device_get_text_extent(textstring)

    def device_get_text_extent(self,textstring):
        return self.device_get_full_text_extent(textstring)

    def get_full_text_extent(self,textstring):
        """
            Calls device specific text extent method.
        """
        return self.device_get_full_text_extent(textstring)

    def device_get_full_text_extent(self,textstring):
        return (0.0, 0.0, 0.0, 0.0)
        #raise NotImplementedError("device_get_full_text_extent() is not implemented")
        #ttm = self.get_text_matrix()
        #ctm = self.get_ctm()  # not device_ctm!!
        #m   = affine.concat( ctm, ttm )
        #ft_engine.transform( affine.affine_params( m )[0:4] )
        #f = self.state.font   ### TEMPORARY ###
        #ft_engine.select_font( f.name, f.size, f.style, f.encoding )   ### TEMPORARY ###
        ##ft_engine.select_font( 'Arial', 10 )   ### TEMPORARY ###
        #ft_engine.antialias( self.state.antialias )
        #glyphs = ft_engine.render( textstring )
        #dy, dx = shape( glyphs.img )
        #return ( dx, dy, -glyphs.bbox[1], 0 )


    #----------------------------------------------------------------
    # Extra routines that aren't part of DisplayPDF
    #
    # Some access to font metrics are needed for laying out text.
    # Not sure how to handle this yet.  The candidates below are
    # from Piddle.  Perhaps there is another alternative?
    #
    #----------------------------------------------------------------



    #def font_height(self):
    #    '''Find the total height (ascent + descent) of the given font.'''
    #    #return self.font_ascent() + self.font_descent()

    #def font_ascent(self):
    #    '''Find the ascent (height above base) of the given font.'''
    #    pass

    #def font_descent(self):
    #    '''Find the descent (extent below base) of the given font.'''
    #    extents = self.dc.GetFullTextExtent(' ', wx_font)
    #    return extents[2]

