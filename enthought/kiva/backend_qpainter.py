#------------------------------------------------------------------------------
# Copyright (c) 2010, Enthought, Inc
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#------------------------------------------------------------------------------
""" This is the QPainter backend for kiva. """


# These are the symbols that a backend has to define.
__all__ = ["GraphicsContext", "Canvas", "CompiledPath",
           "font_metrics_provider"]

from functools import partial
from itertools import izip
import numpy as np
import warnings

# Major package imports.
from enthought.qt.api import QtCore, QtGui

# Local imports.
from fonttools import Font
import constants

cap_style = {}
cap_style[constants.CAP_ROUND]  = QtCore.Qt.RoundCap
cap_style[constants.CAP_SQUARE] = QtCore.Qt.SquareCap
cap_style[constants.CAP_BUTT]   = QtCore.Qt.FlatCap

join_style = {}
join_style[constants.JOIN_ROUND] = QtCore.Qt.RoundJoin
join_style[constants.JOIN_BEVEL] = QtCore.Qt.BevelJoin
join_style[constants.JOIN_MITER] = QtCore.Qt.MiterJoin

draw_modes = {}
draw_modes[constants.FILL]            = QtCore.Qt.OddEvenFill
draw_modes[constants.EOF_FILL]        = QtCore.Qt.WindingFill
draw_modes[constants.STROKE]          = 0
draw_modes[constants.FILL_STROKE]     = QtCore.Qt.OddEvenFill
draw_modes[constants.EOF_FILL_STROKE] = QtCore.Qt.WindingFill

gradient_coord_modes = {}
gradient_coord_modes['userSpaceOnUse'] = QtGui.QGradient.LogicalMode
gradient_coord_modes['objectBoundingBox'] = QtGui.QGradient.ObjectBoundingMode

gradient_spread_modes = {}
gradient_spread_modes['pad'] = QtGui.QGradient.PadSpread
gradient_spread_modes['repeat'] = QtGui.QGradient.RepeatSpread
gradient_spread_modes['reflect'] = QtGui.QGradient.ReflectSpread


class GraphicsContext(object):
    """ Simple wrapper around a Qt QPainter object.
    """
    def __init__(self, size, pix_format="", parent=None, bottom_up=0):
        self._width = size[0]
        self._height = size[1]
        
        self.text_pos = [0.0, 0.0]
        
        # create some sort of device context
        if parent is None:
            self.qt_dc = QtGui.QPixmap(*size)
        else:
            self.qt_dc = parent
        
        self.gc = QtGui.QPainter(self.qt_dc)
        self.path = QtGui.QPainterPath()
        
        # flip y
        trans = QtGui.QTransform()
        trans.translate(0, size[1])
        trans.scale(1.0, -1.0)
        self.gc.setWorldTransform(trans)
        
        # enable antialiasing
        self.gc.setRenderHints(QtGui.QPainter.Antialiasing|QtGui.QPainter.TextAntialiasing,
                               True)
        # set the pen and brush to useful defaults
        self.gc.setPen(QtCore.Qt.black)
        self.gc.setBrush(QtGui.QBrush(QtCore.Qt.SolidPattern))
    
    def __del__(self):
        # stop the painter when drawing to a pixmap
        if isinstance(self.qt_dc, QtGui.QPixmap):
            self.gc.end()

    #----------------------------------------------------------------
    # Size info
    #----------------------------------------------------------------

    def height(self):
        """ Returns the height of the context.
        """
        return self._height
    
    def width(self):
        """ Returns the width of the context.
        """
        return self._width
    
    #----------------------------------------------------------------
    # Coordinate Transform Matrix Manipulation
    #----------------------------------------------------------------
    
    def scale_ctm(self, sx, sy):
        """ Set the coordinate system scale to the given values, (sx,sy).

            sx:float -- The new scale factor for the x axis
            sy:float -- The new scale factor for the y axis            
        """
        self.gc.scale(sx, sy)
        
    def translate_ctm(self, tx, ty):
        """ Translate the coordinate system by the given value by (tx,ty)

            tx:float --  The distance to move in the x direction
            ty:float --   The distance to move in the y direction
        """        
        self.gc.translate(tx, ty)

    def rotate_ctm(self, angle):
        """ Rotates the coordinate space for drawing by the given angle.

            angle:float -- the angle, in radians, to rotate the coordinate 
                           system
        """        
        self.gc.rotate(np.rad2deg(angle))
    
    def concat_ctm(self, transform):
        """ Concatenate the transform to current coordinate transform matrix.
        
            transform:affine_matrix -- the transform matrix to concatenate with
                                       the current coordinate matrix.
        """
        self.gc.setTransform(QtGui.QTransform(transform), True)
    
    def get_ctm(self):
        """ Return the current coordinate transform matrix.  
        
            XXX: This should really return a 3x3 matrix (or maybe an affine
                 object?) like the other API's.  Needs thought.
        """           
        return self.gc.transform()
        
    #----------------------------------------------------------------
    # Save/Restore graphics state.
    #----------------------------------------------------------------

    def save_state(self):
        """ Save the current graphic's context state.
       
            This should always be paired with a restore_state
        """
        self.gc.save()
    
    def restore_state(self):
        """ Restore the previous graphics state.
        """
        self.gc.restore()


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
        """ Set/Unset antialiasing for bitmap graphics context.
        """
        self.gc.setRenderHints(QtGui.QPainter.Antialiasing|QtGui.QPainter.TextAntialiasing,
                               value)
        
    def set_line_width(self,width):
        """ Set the line width for drawing
        
            width:float -- The new width for lines in user space units.
        """
        pen = self.gc.pen()
        pen.setWidthF(width)
        self.gc.setPen(pen)

    def set_line_join(self,style):
        """ Set style for joining lines in a drawing.
            
            style:join_style -- The line joining style.  The available 
                                styles are JOIN_ROUND, JOIN_BEVEL, JOIN_MITER.
        """    
        try:
            sjoin = join_style[style]
        except KeyError:            
            msg = "Invalid line join style.  See documentation for valid styles"
            raise ValueError, msg
        
        pen = self.gc.pen()
        pen.setJoinStyle(sjoin)
        self.gc.setPen(pen)
        
    def set_miter_limit(self,limit):
        """ Specifies limits on line lengths for mitering line joins.
        
            If line_join is set to miter joins, the limit specifies which
            line joins should actually be mitered.  If lines aren't mitered,
            they are joined with a bevel.  The line width is divided by
            the length of the miter.  If the result is greater than the
            limit, the bevel style is used.
            
            limit:float -- limit for mitering joins.
        """
        pen = self.gc.pen()
        pen.setMiterLimit(limit)
        self.gc.setPen(pen)
        
    def set_line_cap(self,style):
        """ Specify the style of endings to put on line ends.
                    
            style:cap_style -- the line cap style to use. Available styles 
                               are CAP_ROUND,CAP_BUTT,CAP_SQUARE
        """    
        try:
            scap = cap_style[style]
        except KeyError:            
            msg = "Invalid line cap style.  See documentation for valid styles"
            raise ValueError, msg
        
        pen = self.gc.pen()
        pen.setCapStyle(scap)
        self.gc.setPen(pen)
       
    def set_line_dash(self,lengths,phase=0):
        """
        
            lengths:float array -- An array of floating point values 
                                   specifing the lengths of on/off painting
                                   pattern for lines.
            phase:float -- Specifies how many units into dash pattern
                           to start.  phase defaults to 0.
        """
        lengths = list(lengths) if lengths is not None else []
        pen = self.gc.pen()
        pen.setDashPattern(lengths)
        pen.setDashOffset(phase)
        self.gc.setPen(pen)

    def set_flatness(self,flatness):
        """ Not implemented
            
            It is device dependent and therefore not recommended by
            the PDF documentation.
        """    
        raise NotImplementedError

    #----------------------------------------------------------------
    # Sending drawing data to a device
    #----------------------------------------------------------------

    def flush(self):
        """ Send all drawing data to the destination device.
        """
        pass
        
    def synchronize(self):
        """ Prepares drawing data to be updated on a destination device.
        """
        pass
    
    #----------------------------------------------------------------
    # Page Definitions
    #----------------------------------------------------------------
    
    def begin_page(self):
        """ Create a new page within the graphics context.
        """
        pass
        
    def end_page(self):
        """ End drawing in the current page of the graphics context.
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
        """ Clear the current drawing path and begin a new one.
        """
        self.path = QtGui.QPainterPath()

    def move_to(self,x,y):    
        """ Start a new drawing subpath at place the current point at (x,y).
        """
        self.path.moveTo(x,y)

    def line_to(self,x,y):
        """ Add a line from the current point to the given point (x,y).
        
            The current point is moved to (x,y).
        """
        self.path.lineTo(x,y)

    def lines(self,points):
        """ Add a series of lines as a new subpath.  
        
            Currently implemented by calling line_to a zillion times.
        
            Points is an Nx2 array of x,y pairs.
        """
        self.path.moveTo(points[0][0],points[0][1])
        for x,y in points[1:]:
            self.path.lineTo(x,y)
    
    def line_set(self, starts, ends):
        """ Draw multiple disjoint line segments.
        """
        for start, end in izip(starts, ends):
            self.path.moveTo(start[0], start[1])
            self.path.lineTo(end[0], end[1])
    
    def rect(self,x,y,sx,sy):
        """ Add a rectangle as a new subpath.
        """
        self.path.addRect(x,y,sx,sy)
    
    def rects(self,rects):
        """ Add multiple rectangles as separate subpaths to the path.
        
            Currently implemented by calling rect a zillion times.
                   
        """
        for x,y,sx,sy in rects:
            self.path.addRect(x,y,sx,sy)
    
    def draw_rect(self, rect, mode=constants.FILL_STROKE):
        """ Draw a rect.
        """
        rect = QtCore.QRectF(*rect)
        if mode == constants.STROKE:
            self.gc.drawRect(rect)
        elif mode in [constants.FILL, constants.EOF_FILL]:
            self.gc.fillRect(rect, self.gc.brush())
        else:
            self.gc.fillRect(rect, self.gc.brush())
            self.gc.drawRect(rect)

    def add_path(self, path):
        """ Add a subpath to the current path.
        """
        self.path.addPath(path.path)

    def close_path(self):
        """ Close the path of the current subpath.
        """
        self.path.closeSubpath()

    def curve_to(self, cp1x, cp1y, cp2x, cp2y, x, y):
        """ 
        """
        self.path.cubicTo(cp1x, cp1y, cp2x, cp2y, x, y)
        
    def quad_curve_to(self,cpx,cpy,x,y):
        """
        """
        self.path.quadTo(cpx, cpy, x, y)
    
    def arc(self, x, y, radius, start_angle, end_angle, clockwise=False):
        """
        """
        sweep_angle = end_angle-start_angle if not clockwise else start_angle-end_angle
        self.path.moveTo(x, y)
        self.path.arcTo(QtCore.QRectF(x-radius, y-radius, radius*2, radius*2),
                        np.rad2deg(start_angle), np.rad2deg(sweep_angle))
    
    def arc_to(self, x1, y1, x2, y2, radius):
        """
        """
        pass

    #----------------------------------------------------------------
    # Getting infomration on paths
    #----------------------------------------------------------------

    def is_path_empty(self):
        """ Test to see if the current drawing path is empty
        """
        return self.path.isEmpty()          
        
    def get_path_current_point(self):
        """ Return the current point from the graphics context.
        
            Note: This should be a tuple or array.
        
        """
        result = self.path.pointAtPercent(1.0)
        return (result.x(), result.y())
            
    def get_path_bounding_box(self):
        """
            should return a tuple or array instead of a strange object.
        """
        result = self.path.boundingRect()
        return (result.x(), result.y(), result.width(), result.height())

    #----------------------------------------------------------------
    # Clipping path manipulation
    #----------------------------------------------------------------

    def clip(self):
        """
        """
        self.gc.setClipPath(self.path)
        
    def even_odd_clip(self):
        """
        """
        self.gc.setClipPath(self.path, operation=QtCore.Qt.IntersectClip)
        
    def clip_to_rect(self, x, y, w, h):
        """ Clip context to the given rectangular region.
        
            Region should be a 4-tuple or a sequence.            
        """
        self.gc.setClipRect(QtCore.QRectF(x,y,w,h))
        
    def clip_to_rects(self):
        """
        """
        msg = "clip_to_rects not implemented on Qt yet."
        raise NotImplementedError, msg
        
    #----------------------------------------------------------------
    # Color space manipulation
    #
    # I'm not sure we'll mess with these at all.  They seem to
    # be for setting the color system.  Hard coding to RGB or
    # RGBA for now sounds like a reasonable solution.
    #----------------------------------------------------------------

    def set_fill_color_space(self):
        """
        """
        msg = "set_fill_color_space not implemented on Qt yet."
        raise NotImplementedError, msg
    
    def set_stroke_color_space(self):
        """
        """
        msg = "set_stroke_color_space not implemented on Qt yet."
        raise NotImplementedError, msg
        
    def set_rendering_intent(self):
        """
        """
        msg = "set_rendering_intent not implemented on Qt yet."
        raise NotImplementedError, msg
        
    #----------------------------------------------------------------
    # Color manipulation
    #----------------------------------------------------------------

    def set_fill_color(self, color):
        """
        """
        r,g,b = color[:3]
        try:
            a = color[3]
        except IndexError:
            a = 1.0
        brush = self.gc.brush()
        brush.setColor(QtGui.QColor.fromRgbF(r,g,b,a))
        self.gc.setBrush(brush)
    
    def set_stroke_color(self, color):
        """
        """
        r,g,b = color[:3]
        try:
            a = color[3]
        except IndexError:
            a = 1.0
        pen = self.gc.pen()
        pen.setColor(QtGui.QColor.fromRgbF(r,g,b,a))
        self.gc.setPen(pen)
    
    def set_alpha(self, alpha):
        """
        """
        self.gc.setOpacity(alpha)

    #----------------------------------------------------------------
    # Gradients
    #----------------------------------------------------------------
    
    def _apply_gradient(self, grad, stops, spread_method, units):
        """ Configures a gradient object and sets it as the current brush.
        """
        grad.setSpread(gradient_spread_modes[spread_method])
        grad.setCoordinateMode(gradient_coord_modes[units])
        
        for stop in stops:
            grad.setColorAt(stop[0], QtGui.QColor.fromRgbF(*stop[1:]))
        
        self.gc.setBrush(QtGui.QBrush(grad))
    
    def linear_gradient(self, x1, y1, x2, y2, stops, spread_method,
                        units='userSpaceOnUse'):
        """ Sets a linear gradient as the current brush.
        """
        grad = QtGui.QLinearGradient(x1, y1,x2, y2)
        self._apply_gradient(grad, stops, spread_method, units)

    def radial_gradient(self, cx, cy, r, fx, fy, stops, spread_method,
                        units='userSpaceOnUse'):
        """ Sets a radial gradient as the current brush.
        """
        grad = QtGui.QRadialGradient(cx, cy, r, fx, fy)
        self._apply_gradient(grad, stops, spread_method, units)
    
    #----------------------------------------------------------------
    # Drawing Images
    #----------------------------------------------------------------
        
    def draw_image(self, img, rect=None):
        """
        img is either a N*M*3 or N*M*4 numpy array, or a Kiva image

        rect - a tuple (x,y,w,h)
        """
        from enthought.kiva import agg
        
        def copy_padded(array):
            "pad image rows to multiples of 4 pixels"
            y,x,d = array.shape
            pad = 4-(x%4)
            if pad == 4:
                return array
            ret = np.zeros((y,x+pad,d), dtype=np.uint8)
            ret[:,:x] = array[:]
            return ret

        if type(img) == type(np.array([])):
            # Numeric array
            if img.shape[2]==3:
                format = QtGui.QImage.Format_RGB888
            elif img.shape[2]==4:
                format = QtGui.QImage.Format_RGB32
            width, height = img.shape[:2]
            copy_array = copy_padded(img)
            draw_img = QtGui.QImage(img.astype(np.uint8), copy_array.shape[1],
                                    height, format)
            pixmap = QtGui.QPixmap.fromImage(draw_img)
        elif isinstance(img, agg.GraphicsContextArray):
            converted_img = img.convert_pixel_format('bgra32', inplace=0)
            copy_array = copy_padded(converted_img.bmp_array)
            width, height = img.width(), img.height()
            draw_img = QtGui.QImage(copy_array.flatten(),
                                    copy_array.shape[1], height, QtGui.QImage.Format_RGB32)
            pixmap = QtGui.QPixmap.fromImage(draw_img)
        elif (isinstance(img, GraphicsContext) and
              isinstance(img.gc.device(), QtGui.QPixmap)):
            # An offscreen Qt kiva context
            pixmap = img.gc.device()
            width, height = pixmap.width(), pixmap.height()
        else:
            warnings.warn("Cannot render image of type '%r' into Qt4 context." % \
                    type(img))
            return
        
        # create a rect object to draw into
        if rect is None:
            dest_rect = QtCore.QRectF(0.0, 0.0, self.width(), self.height())
        else:
            dest_rect = QtCore.QRectF(*rect)
        
        # draw using the entire image's data
        source_rect = QtCore.QRectF(0.0, 0.0, width, height)
        
        flip_trans = QtGui.QTransform()
        flip_trans.scale(1.0, -1.0)
        pixmap = pixmap.transformed(flip_trans)
        
        # draw
        self.gc.drawPixmap(dest_rect, pixmap, source_rect)

    #----------------------------------------------------------------
    # Drawing Text
    #----------------------------------------------------------------
    
    def select_font(self, name, size, textEncoding):
        """ Set the font for the current graphics context.
        """
        self.gc.setFont(QtGui.QFont(name, size))

    def set_font(self, font):
        """ Set the font for the current graphics context.
        """
        self.select_font(font.face_name, font.size, None)
    
    def set_font_size(self, size):
        """
        """
        font = self.gc.font()
        font.setPointSizeF(size)
        self.gc.setFont(font)
        
    def set_character_spacing(self, spacing):
        """
        """
        font = self.gc.font()
        font.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, spacing)
        self.gc.setFont(font)
            
    def set_text_drawing_mode(self):
        """
        """
        pass
    
    def set_text_position(self,x,y):
        """
        """
        self.text_pos = [x,y]
        
    def get_text_position(self):
        """
        """
        return self.text_pos
        
    def set_text_matrix(self,ttm):
        """
        """
        msg = "Text matrix not available on Qt yet."
        raise NotImplementedError, msg
        
    def get_text_matrix(self):
        """
        """        
        msg = "Text matrix not available on Qt yet."
        raise NotImplementedError, msg
        
    def show_text(self, text, point=None):
        """ Draw text on the device at current text position.
            
            This is also used for showing text at a particular point
            specified by x and y.
        """
        if point is None:
            pos = tuple(self.text_pos)
        else:
            pos = tuple(point)
        
        unflip_trans = QtGui.QTransform()
        unflip_trans.scale(1.0, -1.0)
        
        self.gc.save()
        self.gc.setTransform(unflip_trans, True)
        self.gc.drawText(QtCore.QPointF(*pos), text)
        self.gc.restore()
        
    def show_text_at_point(self, text, x, y):
        """ Draw text at some point (x,y).
        """
        self.show_text(text, (x,y))

    def show_glyphs(self):
        """
        """
        msg = "show_glyphs not implemented on Qt yet."
        raise NotImplementedError, msg
    
    def get_text_extent(self, text):
        """ Returns the bounding rect of the rendered text
        """
        fm = self.gc.fontMetrics()
        rect = fm.boundingRect(text)
        
        return rect.left(), -fm.descent(), rect.right(), fm.height()

    def get_full_text_extent(self, text):
        """ Backwards compatibility API over .get_text_extent() for Enable
        """
        x1, y1, x2, y2 = self.get_text_extent(text)
        
        return x2, y2, y1, x1
    
    #----------------------------------------------------------------
    # Painting paths (drawing and filling contours)
    #----------------------------------------------------------------

    def stroke_path(self):
        """
        """
        self.gc.strokePath(self.path, self.gc.pen())
        self.begin_path()
    
    def fill_path(self):
        """
        """
        self.gc.fillPath(self.path, self.gc.brush())
        self.begin_path()
        
    def eof_fill_path(self):
        """
        """
        self.path.setFillRule(QtCore.Qt.OddEvenFill)
        self.gc.fillPath(self.path, self.gc.brush())
        self.begin_path()

    def stroke_rect(self,rect):
        """
        """
        self.gc.drawRect(QtCore.QRectF(*rect))
    
    def stroke_rect_with_width(self,rect,width):
        """
        """
        save_pen = self.gc.pen()
        draw_pen = QtGui.QPen(save_pen)
        draw_pen.setWidthF(width)
        
        self.gc.setPen(draw_pen)
        self.stroke_rect(rect)
        self.gc.setPen(save_pen)

    def fill_rect(self,rect):
        """
        """
        self.gc.fillRect(QtCore.QRectF(*rect), self.gc.brush())
        
    def fill_rects(self):
        """
        """
        msg = "fill_rects not implemented on Qt yet."
        raise NotImplementedError, msg
    
    def clear_rect(self, rect):
        """
        """
        self.gc.eraseRect(QtCore.QRectF(*rect))
    
    def clear(self, clear_color=(1.0,1.0,1.0,1.0)):
        """
        """
        if len(clear_color) == 4:
            r,g,b,a = clear_color
        else:
            r,g,b = clear_color
            a = 1.0
        self.gc.setBackground(QtGui.QBrush(QtGui.QColor.fromRgbF(r,g,b,a)))
        self.gc.eraseRect(QtCore.QRectF(0,0,self.width(),self.height()))
    
    def draw_path(self, mode=constants.FILL_STROKE):
        """ Walk through all the drawing subpaths and draw each element.
        
            Each subpath is drawn separately.
        """
        if mode == constants.STROKE:
            self.stroke_path()
        elif mode in [constants.FILL, constants.EOF_FILL]:
            mode = draw_modes[mode]
            self.path.setFillRule(mode)
            self.fill_path()
        else:
            mode = draw_modes[mode]
            self.path.setFillRule(mode)
            self.gc.drawPath(self.path)
        self.begin_path()
    
    def get_empty_path(self):
        """ Return a path object that can be built up and then reused.
        """
        return CompiledPath()
    
    def draw_path_at_points(self, points, path, mode=constants.FILL_STROKE):
        # set up drawing state and function
        if mode == constants.STROKE:
            draw_func = partial(self.gc.strokePath, path.path, self.gc.pen())
        elif mode in [constants.FILL, constants.EOF_FILL]:
            mode = draw_modes[mode]
            path.path.setFillRule(mode)
            draw_func = partial(self.gc.fillPath, path.path, self.gc.brush())
        else:
            mode = draw_modes[mode]
            path.path.setFillRule(mode)
            draw_func = partial(self.gc.drawPath, path.path)
        
        for point in points:
            x, y = point
            self.gc.save()
            self.gc.translate(x, y)
            draw_func()
            self.gc.restore()


class CompiledPath(object):
    def __init__(self):
        self.path = QtGui.QPainterPath()

    def begin_path(self):
        return

    def move_to(self, x, y):
        self.path.moveTo(x, y)

    def arc(self, x, y, r, start_angle, end_angle, clockwise=False):
        sweep_angle = end_angle-start_angle if not clockwise else start_angle-end_angle
        self.path.moveTo(x, y)
        self.path.arcTo(QtCore.QRectF(x-r, y-r, r*2, r*2),
                        np.rad2deg(start_angle), np.rad2deg(sweep_angle))

    def arc_to(self, x1, y1, x2, y2, r):
        pass

    def curve_to(self, cx1, cy1, cx2, cy2, x, y):
        self.path.cubicTo(cx1, cy1, cx2, cy2, x, y)

    def line_to(self, x, y):
        self.path.lineTo(x, y)

    def lines(self, points):
        self.path.moveTo(points[0][0],points[0][1])
        for x,y in points[1:]:
            self.path.lineTo(x,y)

    def add_path(self, other_path):
        if isinstance(other_path, CompiledPath):
            self.path.addPath(other_path.path)

    def quad_curve_to(self, cx, cy, x, y):
        self.path.quadTo(cx, cy, x, y)

    def rect(self, x, y, sx, sy):
        self.path.addRect(x, y, sx, sy)

    def rects(self, rects):
        for x,y,sx,sy in rects:
            self.path.addRect(x,y,sx,sy)

    def close_path(self):
        self.path.closeSubpath()

    def is_empty(self):
        return self.path.isEmpty()

    def get_current_point(self):
        point = self.path.pointAtPercent(1.0)
        return point.x(), point.y()

    def get_bounding_box(self):
        rect = self.path.boundingRect(self.path)
        return rect.x(), rect.y(), rect.width(), rect.height()


class Canvas(QtGui.QWidget):
    def __init__(self, *args):
        QtGui.QWidget.__init__(self, *args)

        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

        self.clear_color = (1,1,1)

    def do_draw(self, gc):
        """ Method to be implemented by subclasses to actually perform various
        GC drawing commands before the GC is blitted into the screen.
        """
        pass

    def paintEvent(self, event):
        gc = self.new_gc(size=None)
        # Clear the gc.
        gc.clear(self.clear_color)

        # Draw on the GC.
        self.do_draw(gc)

    def new_gc(self, size):
        """ Creates a new GC of the requested size (or of the widget's current
        size if size is None) and stores it in self.gc.
        """
        if size is None:
            width = self.width()
            height = self.height()
        else:
            width, height = size

        return GraphicsContext((width, height), parent=self)


def font_metrics_provider():
    """ Creates an object to be used for querying font metrics.
    """
    return GraphicsContext((1,1))


