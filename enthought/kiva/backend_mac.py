#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# some parts copyright 2002 by Space Telescope Science Institute
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#------------------------------------------------------------------------------
""" Mac version Kiva's core2d drawing API

    This is a wrapper around the Mac Quartz API.  Much of Kiva's
    core2d API maps one-to-one with the Quartz API.  As a result, their
    is no need to inherit from the GraphicsContextBase class defined in
    basecore2d.py.
"""
import copy
from Carbon import CG
from Carbon import CoreGraphics
import basecore2d
import constants
from constants import *

cap_style = {}
cap_style[CAP_ROUND]  = CoreGraphics.kCGLineCapRound
cap_style[CAP_SQUARE] = CoreGraphics.kCGLineCapSquare
cap_style[CAP_BUTT]   = CoreGraphics.kCGLineCapButt

join_style = {}
join_style[JOIN_ROUND] = CoreGraphics.kCGLineJoinRound
join_style[JOIN_BEVEL] = CoreGraphics.kCGLineJoinBevel
join_style[JOIN_MITER] = CoreGraphics.kCGLineJoinMiter

draw_modes = {}
draw_modes[FILL]            = CoreGraphics.kCGPathFill
draw_modes[EOF_FILL]        = CoreGraphics.kCGPathEOFill
draw_modes[STROKE]          = CoreGraphics.kCGPathStroke
draw_modes[FILL_STROKE]     = CoreGraphics.kCGPathFillStroke
draw_modes[EOF_FILL_STROKE] = CoreGraphics.kCGPathEOFillStroke

CompiledPath = None

class GraphicsContext(object):
    """ Simple wrapper around a standard mac graphics context.
    """ 
    def __init__(self,gc):
        self.corner_pixel_origin = True
        self.gc = gc
    #----------------------------------------------------------------
    # Coordinate Transform Matrix Manipulation
    #----------------------------------------------------------------
    
    def scale_ctm(self, sx, sy):
        """ Set the coordinate system scale to the given values, (sx,sy).

            sx:float -- The new scale factor for the x axis
            sy:float -- The new scale factor for the y axis            
        """
        self.gc.CGContextScaleCTM(sx, sy)
        
    def translate_ctm(self, tx, ty):
        """ Translate the coordinate system by the given value by (tx,ty)

            tx:float --  The distance to move in the x direction
            ty:float --   The distance to move in the y direction
        """        
        self.gc.CGContextTranslateCTM( tx, ty)

    def rotate_ctm(self, angle):
        """ Rotates the coordinate space for drawing by the given angle.

            angle:float -- the angle, in radians, to rotate the coordinate 
                           system
        """        
        self.gc.CGContextRotateCTM( angle)
    
    def concat_ctm(self, transform):
        """ Concatenate the transform to current coordinate transform matrix.
        
            transform:affine_matrix -- the transform matrix to concatenate with
                                       the current coordinate matrix.
        """
        self.gc.CGContextConcatCTM(transform)
    
    def get_ctm(self):
        """ Return the current coordinate transform matrix.  
        
            XXX: This should really return a 3x3 matrix (or maybe an affine
                 object?) like the other API's.  Needs thought.
        """           
        return self.gc.CGContextGetCTM()
        
    #----------------------------------------------------------------
    # Save/Restore graphics state.
    #----------------------------------------------------------------

    def save_state(self):
        """ Save the current graphic's context state.
       
            This should always be paired with a restore_state
        """    
        self.gc.CGContextSaveGState()
    
    def restore_state(self):
        """ Restore the previous graphics state.
        """
        self.gc.CGContextRestoreGState()


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
    
    def set_should_antialias(self,value):
        """ Set/Unset antialiasing for bitmap graphics context.
        """
        self.gc.CGContextSetShouldAntialias(value)
        
    def set_line_width(self,width):
        """ Set the line width for drawing
        
            width:float -- The new width for lines in user space units.
        """
        self.gc.CGContextSetLineWidth(width)

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
        self.gc.CGContextSetLineJoin(sjoin)
        
    def set_miter_limit(self,limit):
        """ Specifies limits on line lengths for mitering line joins.
        
            If line_join is set to miter joins, the limit specifies which
            line joins should actually be mitered.  If lines aren't mitered,
            they are joined with a bevel.  The line width is divided by
            the length of the miter.  If the result is greater than the
            limit, the bevel style is used.
            
            limit:float -- limit for mitering joins.
        """
        self.gc.CGContextSetMiterLimit(limit)
        
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
        self.gc.CGContextSetLineCap(scap)
       
    def set_line_dash(self,lengths,phase=0):
        """
        
            lengths:float array -- An array of floating point values 
                                   specifing the lengths of on/off painting
                                   pattern for lines.
            phase:float -- Specifies how many units into dash pattern
                           to start.  phase defaults to 0.
        """
        #raise NotImplementedError, "line dash is not implemented on Mac OS X yet."
        pass
    def set_flatness(self,flatness):
        """ Not implemented
            
            It is device dependent and therefore not recommended by
            the PDF documentation.
        """    
        self.gc.CGContextSetFlatness( flatness)

    #----------------------------------------------------------------
    # Sending drawing data to a device
    #----------------------------------------------------------------

    def flush(self):
        """ Send all drawing data to the destination device.
        """
        self.gc.CGContextFlush()
        
    def synchronize(self):
        """ Prepares drawing data to be updated on a destination device.
        """
        self.gc.ContextSynchronize()
    
    #----------------------------------------------------------------
    # Page Definitions
    #----------------------------------------------------------------
    
    def begin_page(self):
        """ Create a new page within the graphics context.
        """
        self.gc.CGContextBeginPage()
        
    def end_page(self):
        """ End drawing in the current page of the graphics context.
        """        
        self.gc.CGContextEndPage()        
    
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
        self.gc.CGContextBeginPath()

    def move_to(self,x,y):    
        """ Start a new drawing subpath at place the current point at (x,y).
        """
        self.gc.CGContextMoveToPoint(x,y)
        
    def line_to(self,x,y):
        """ Add a line from the current point to the given point (x,y).
        
            The current point is moved to (x,y).
        """
        self.gc.CGContextAddLineToPoint(x,y)
            
    def lines(self,points):
        """ Add a series of lines as a new subpath.  
        
            Currently implemented by calling line_to a zillion times.
        
            Points is an Nx2 array of x,y pairs.
            
            current_point is moved to the last point in points           
        """
        self.gc.CGContextMoveToPoint(points[0][0],points[0][1])
        for x,y in points[1:]:
            self.gc.CGContextAddLineToPoint(x,y)
                
    def rect(self,x,y,sx,sy):
        """ Add a rectangle as a new subpath.
        """
        self.gc.CGContextAddRect((x,y,sx,sy))
    
    def rects(self,rects):
        """ Add multiple rectangles as separate subpaths to the path.
        
            Currently implemented by calling rect a zillion times.
                   
        """
        for x,y,sx,sy in rects:
            self.gc.CGContextAddRect((x,y,sx,sy))
        
    def close_path(self):
        """ Close the path of the current subpath.
        """
        self.gc.CGContextClosePath()

    def curve_to(self, cp1x, cp1y, cp2x, cp2y, x, y):
        """ 
        """
        self.gc.CGContextAddCurveToPoint( cp1x, cp1y, cp2x, cp2y, x, y )
        
    def quad_curve_to(self,cpx,cpy,x,y):
        """
        """
        self.gc.CGContextAddQuadCurveToPoint( cpx, cpy, x, y)
    
    def arc(self, x, y, radius, start_angle, end_angle, clockwise):
        """
        """
        self.gc.CGContextAddArc( x, y, radius, start_angle, end_angle, 
                           clockwise)
    
    def arc_to(self, x1, y1, x2, y2, radius):
        """
        """
        self.gc.CGContextAddArcToPoint( x1, y1, x2, y2, radius)
                                           
    #----------------------------------------------------------------
    # Getting infomration on paths
    #----------------------------------------------------------------

    def is_path_empty(self):
        """ Test to see if the current drawing path is empty
        """
        return self.gc.CGContextIsPathEmpty()          
        
    def get_path_current_point(self):
        """ Return the current point from the graphics context.
        
            Note: This should be a tuple or array.
        
        """
        result = self.gc.CGContextGetPathCurrentPoint(self.gc)
        return translate_to_array(result)
            
    def get_path_bounding_box(self):
        """
            should return a tuple or array instead of a strange object.
        """
        result = self.gc.CGContextGetPathBoundingBox(self.gc)
        return translate_to_array(result)

    #----------------------------------------------------------------
    # Clipping path manipulation
    #----------------------------------------------------------------

    def clip(self):
        """
        """
        self.gc.CGContextClip(self.gc)
        
    def even_odd_clip(self):
        """
        """
        self.gc.CGContextEOClip(self.gc)
        
    def clip_to_rect(self,rect):
        """ Clip context to the given rectangular region.
        
            Region should be a 4-tuple or a sequence.            
        """
        self.gc.CGContextClipToRect( rect)
        
    def clip_to_rects(self):
        """
        """
        msg = "clip_to_rects not implemented on Macintosh yet."
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
        msg = "set_fill_color_space not implemented on Macintosh yet."
        raise NotImplementedError, msg
    
    def set_stroke_color_space(self):
        """
        """
        msg = "set_stroke_color_space not implemented on Macintosh yet."
        raise NotImplementedError, msg
        
    def set_rendering_intent(self):
        """
        """
        msg = "set_rendering_intent not implemented on Macintosh yet."
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
        self.gc.CGContextSetRGBFillColor( r, g, b, a)
    
    def set_stroke_color(self, color):
        """
        """
        r,g,b = color[:3]
        try:
            a = color[3]
        except IndexError:
            a = 1.0
        self.gc.CGContextSetRGBStrokeColor( r, g, b, a)
    
    def set_alpha(self, alpha):
        """
        """
        self.gc.CGContextSetAlpha(alpha)
    
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
        
    def draw_image(self):
        """
        """
        pass
    
    #----------------------------------------------------------------
    # Drawing PDF documents
    #----------------------------------------------------------------

    #def draw_pdf_document(self):
    #    """
    #    """
    #    pass    

    #----------------------------------------------------------------
    # Drawing Text
    #----------------------------------------------------------------
    
    def select_font(self, name, size, textEncoding):
        """
        """
        self.gc.CGContextSelectFont( name, size, textEncoding)

    def set_font(self,font):
        """ Set the font for the current graphics context.
        
            I need to figure out this one.
        """
        raise NotImplementedError, "select_font isn't implemented yet for Macintosh"
    
    def set_font_size(self,size):
        """
        """
        self.gc.CGContextSetFontSize( size)
        
    def set_character_spacing(self):
        """
        """
        pass
            
    def set_text_drawing_mode(self):
        """
        """
        pass
    
    def set_text_position(self,x,y):
        """
        """
        self.gc.CGContextSetTextPosition( x, y)
        
    def get_text_position(self):
        """
        """
        return self.state.text_matrix[2,:2]
        
    def set_text_matrix(self,ttm):
        """
        """
        self.gc.CGContextGetTextMatrix(ttm)
        
    def get_text_matrix(self):
        """
        """        
        self.gc.CGContextGetTextMatrix(self.gc)
        
    def show_text(self, text, x = None, y = None):
        """ Draw text on the device at current text position.
            
            This is also used for showing text at a particular point
            specified by x and y.
        """
        if x is None:
            self.gc.CGContextShowText( text)
        else:            
            self.gc.CGContextShowTextAtPoint(x,y,text)
                   
        
    def show_glyphs(self):
        """
        """
        msg = "show_glyphs not implemented on Macintosh yet."
        raise NotImplementedError, msg
    
    #----------------------------------------------------------------
    # Painting paths (drawing and filling contours)
    #----------------------------------------------------------------

    def stroke_path(self):
        """
        """
        self.gc.CGContextStrokePath()
    

    
    def fill_path(self):
        """
        """
        self.gc.CGContextFillPath()
        
    def eof_fill_path(self):
        """
        """
        self.gc.CGContextEOFillPath()

    def stroke_rect(self,rect):
        """
        """
        self.gc.CGContextStrokeRect(rect)
    
    def stroke_rect_with_width(self,rect,width):
        """
        """
        self.gc.CGContextStrokeRectWithWidth(rect,width)

    def fill_rect(self,rect):
        """
        """
        self.gc.CGContextFillRect(rect)
        
    def fill_rects(self):
        """
        """
        msg = "fill_rects not implemented on Macintosh yet."
        raise NotImplementedError, msg
    
    def clear_rect(self,rect):
        """
        """
        self.gc.CGContextClearRect(rect)
            
    def draw_path(self,mode):
        """ Walk through all the drawing subpaths and draw each element.
        
            Each subpath is drawn separately.
        """
        cg_mode = draw_modes[mode]
        self.gc.CGContextDrawPath(cg_mode)
       
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

import W

class Canvas(W.Widget):
    def __init__(self, parent=None, id=-1,position = None, size = None):
        #---------------------------------------------------------------------
        # Set up position and size of canvas.  The position_size variable
        # is a 4-tuple, (l,t,r,b).  If l and t are positive, they specify
        # the widget's distance from the left and top edge of the window.  
        # If r and b are *negative*, they specify the widgets distance from 
        # the right and bottom of the window.  If they are positive, they 
        # the width and height of the widget respectively.
        #
        # We'll default to pretty much filling the entire window.
        #---------------------------------------------------------------------
        position_size = [10,10,-10,-30]
        if position:
            position_size[:2] = position
        if size:
            position_size[2:] = size
        W.Widget.__init__(self, position_size)
        #self.bind('draw',self._draw)
        #self.set_draw_function(default_draw)

    def size(self):
        return self.getpossize()[-2:]
            
    def client_gc(self):
        """ Create a GraphicsContext object that is setup with the
            origin and clipping boundaries of this widget.
        """
        if not self._bounds:
            return
        l, t, r, b = self._bounds
        width = r - l
        height = b - t
        x, y = l, b
        raw_gc = CG.CreateCGContextForPort(self._parentwindow.wid)
        gc = GraphicsContext(raw_gc)
        wx, wy, wwidth, wheight = self._parentwindow._bounds
        gc.translate_ctm(x, wheight - y)
        self.border_box = (0, 0, width, height)
        gc.clip_to_rect(self.border_box)

        # initialize to some standard font (necessary?)
        gc.select_font("Times New Roman",12,1)

        return gc
    
    def draw(self,gc,vis_rgn=None):
        gc = self.client_gc()
        self.do_draw(gc)
        self.draw_border(gc)

    def draw_border(self, gc):
        gc.stroke_rect_with_width(self.border_box, 0.5)
            
    def do_draw(self,gc,vis_rgn=None):
        pass
    """        
    def draw_old(self, gc, vis_rgn=None):

        if not self._visible:
            return
        
        if not gc:
            print 'oops'
            return 
        gc.save_state()
        #gc.scale_ctm(2,2)
        args = (gc,) + self.draw_args
        self.draw_function(*args,**self.draw_kwargs)
        gc.restore_state()
        self.draw_border(gc)
    
    def set_draw_function(self,function, *args, **kwargs):
        self.draw_function = function
        self.draw_args = args
        self.draw_kwargs = kwargs
        gc = self.client_gc()
        self.draw(gc)  
    """
class CanvasWindow(W.Window):
    def __init__(self, id=-1, title='Drawing Canvas',size=(600,600),
                 canvas_class=Canvas):
        W.Window.__init__(self,size, title)
        self.canvas = canvas_class(self)
        self.open()
    #def __init__(self,id=-1,title="Drawing Canvas", size=(300, 300),**kw):
    #    """ id isn't used on Mac.
    #    """
    #    W.Window.__init__(self,size, title)
    #    self.canvas = Canvas()
    #   self.open()                   

def show_all_samplers():
    # I think this needs to run from within the MacPython IDE.
    import os
    os.environ['KIVA_WISHLIST'] = 'mac'
    import core2d
    reload(core2d)
    import sampler
    reload(sampler)    
    sampler.show_all_samplers(default_size=(600,700))

if __name__ == "__main__":
    #w =test_canvas_window(minsize=(100,100))
    #w.canvas.set_draw_function(cap_sampler)
    show_all_samplers()    
