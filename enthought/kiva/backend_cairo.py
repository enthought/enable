""" Implementation of the core2d drawing library, using cairo for rendering

    :Author:      Bryan Cole (bryan@cole.uklinux.net)
    :Copyright:   Bryan Cole (except parts copied from basecore2d)
    :License:     BSD Style

    This is currently under development and is not yet fully functional.
        
"""


import cairo
import basecore2d
import constants

import copy
from itertools import izip

line_join = {constants.JOIN_BEVEL: cairo.LINE_JOIN_BEVEL,
                  constants.JOIN_MITER: cairo.LINE_JOIN_MITER,
                  constants.JOIN_ROUND: cairo.LINE_JOIN_ROUND
                  }

line_cap = {constants.CAP_BUTT: cairo.LINE_CAP_BUTT,
            constants.CAP_ROUND: cairo.LINE_CAP_ROUND,
            constants.CAP_SQUARE: cairo.LINE_CAP_SQUARE
            }

font_slant = {"regular":cairo.FONT_SLANT_NORMAL,
               "bold":cairo.FONT_SLANT_NORMAL,
                "italic":cairo.FONT_SLANT_ITALIC,
                 "bold italic":cairo.FONT_SLANT_ITALIC}

font_weight = {"regular":cairo.FONT_WEIGHT_NORMAL,
               "bold":cairo.FONT_WEIGHT_BOLD,
                "italic":cairo.FONT_WEIGHT_NORMAL,
                 "bold italic":cairo.FONT_WEIGHT_BOLD}

class GraphicsState(object):
    """ Holds information used by a graphics context when drawing.

        I'm not sure if these should be a separate class, a dictionary,
        or part of the GraphicsContext object.  Making them a dictionary
        or object simplifies save_state and restore_state a little bit.

    """
    def __init__(self):       
        self.fill_color = [0.0,0.0,0.0]
        self.stroke_color = [0.0,0.0,0.0]
        self.alpha = 1.0

    def copy(self):
        return copy.deepcopy(self)

class GraphicsContext(basecore2d.GraphicsContextBase):
    def __init__(self, cairoCtx):
        self._ctx = cairoCtx
        self.state = GraphicsState()
        self.state_stack = []
        
    def scale_ctm(self, sx, sy):
        """ Sets the coordinate system scale to the given values, (sx,sy).

            Parameters
            ----------
            sx : float
                The new scale factor for the x axis
            sy : float 
                The new scale factor for the y axis
        """
        self._ctx.scale(sx, sy)
        
    def translate_ctm(self, tx, ty):
        """ Translates the coordinate system by the value given by (tx,ty)

            Parameters
            ----------
            tx : float 
                The distance to move in the x direction
            ty : float
                The distance to move in the y direction
        """
        self._ctx.translate(tx, ty)
        
    def rotate_ctm(self, angle):
        """ Rotates the coordinate space for drawing by the given angle.

            Parameters
            ----------
            angle : float
                the angle, in radians, to rotate the coordinate system
        """        
        self._ctx.rotate(angle)
        
    def concat_ctm(self, transform):
        """ Concatenates the transform to current coordinate transform matrix.
        
            Parameters
            ----------
            transform : affine_matrix
                the transform matrix to concatenate with
                the current coordinate matrix.
        """
        try:
            #assume transform is a cairo.Matrix object
            self._ctx.transform(transform)
        except TypeError:
            #now assume transform is a list of matrix elements (floats)
            self._ctx.transform(cairo.Matrix(*transform))
            
        
    def get_ctm(self):
        """ Returns the current coordinate transform matrix.
        """           
        return list(self._ctx.get_matrix())
        
    #----------------------------------------------------------------
    # Save/Restore graphics state.
    #----------------------------------------------------------------

    def save_state(self):
        """ Saves the current graphic's context state.
       
            Always pair this with a `restore_state()`.
        """    
        self._ctx.save()
        self.state_stack.append(self.state)
        self.state = self.state.copy()
    
    def restore_state(self):
        """ Restores the previous graphics state.
        """
        self._ctx.restore()
        self.state = self.state_stack.pop()
        
    #----------------------------------------------------------------
    # Manipulate graphics state attributes.
    #----------------------------------------------------------------
    
    def set_antialias(self,value):
        """ Sets/Unsets anti-aliasing for bitmap graphics context.
        
            Ignored on most platforms.    
        """
        if bool(value):
            val = cairo.ANTIALIAS_DEFAULT
        else:
            val = cairo.ANTIALIAS_NONE
        self._ctx.set_antialias(val)
        
    def set_line_width(self,width):
        """ Sets the line width for drawing

            Parameters
            ----------
            width : float
                The new width for lines in user space units.
        """
        self._ctx.set_line_width(width)

    def set_line_join(self,style):
        """ Sets the style for joining lines in a drawing.

            Parameters
            ----------
            style : join_style
                The line joining style.  The available 
                styles are JOIN_ROUND, JOIN_BEVEL, JOIN_MITER.
        """    
        try:
            self._ctx.set_line_join(line_join[style])
        except KeyError:
            raise ValueError("Invalid line-join style")
        
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
        self._ctx.set_miter_limit(limit)
        
    def set_line_cap(self,style):
        """ Specifies the style of endings to put on line ends.

            Parameters
            ----------
            style : cap_style
                The line cap style to use. Available styles 
                are CAP_ROUND, CAP_BUTT, CAP_SQUARE.
        """    
        try:
            self._ctx.set_line_cap(line_cap[style])
        except KeyError:
            raise ValueError("Invalid line cap style")
       
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
        if pattern is not None:
            pattern = list(pattern)                    
            self._ctx.set_dash(pattern, phase)
        
    def set_flatness(self,flatness):
        """ Not implemented
            
            It is device dependent and therefore not recommended by
            the PDF documentation.
            
            flatness determines how accurately lines are rendered.  Setting it
            to values less than one will result in more accurate drawings, but
            they take longer.  It defaults to None
        """    
        self._ctx.set_tolerance(flatness)

    #----------------------------------------------------------------
    # Sending drawing data to a device
    #----------------------------------------------------------------

    def flush(self):
        """ Sends all drawing data to the destination device.
        
            Currently this is a NOP for wxPython.
        """
        s = self._ctx.get_target()
        s.flush()
        
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
        self._ctx.new_path()

    def move_to(self,x,y):    
        """ Starts a new drawing subpath and place the current point at (x,y).
        
            Notes:
                Not sure how to treat state.current_point.  Should it be the
                value of the point before or after the matrix transformation?
                It looks like before in the PDF specs.
        """        
        self._ctx.move_to(x,y)
        
    def line_to(self,x,y):
        """ Adds a line from the current point to the given point (x,y).
        
            The current point is moved to (x,y).
            
            What should happen if move_to hasn't been called? Should it always 
            begin at 0,0 or raise an error?
            
            Notes:
                See note in move_to about the current_point.
        """
        print "line", x, y
        self._ctx.line_to(x,y)
    
    def lines(self,points):
        """ Adds a series of lines as a new subpath.  

            Parameters
            ----------
            
            points 
                an Nx2 array of x,y pairs
            
            The current_point is moved to the last point in 'points'
        """        
        self._ctx.new_sub_path()
        for point in points:
            self._ctx.line_to(*point)

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
            
            N.B. Cairo cannot make disjointed lines as a single subpath, 
            thus each line forms it's own subpath
        """
        for start, end in izip(starts, ends):
            self._ctx.move_to(*start)
            self._ctx.line_to(*end)
        
    def rect(self,x,y,sx,sy):
        """ Adds a rectangle as a new subpath.
        """
        self._ctx.rectangle(x,y,sx,sy)

#    def draw_rect(self, rect, mode):
#        self.rect(*rect)
#        self.draw_path(mode=mode)
#
#    def rects(self,rects):
#        """ Adds multiple rectangles as separate subpaths to the path.
#        
#            Not very efficient -- calls rect multiple times.
#        """
#        for x,y,sx,sy in rects:
#            self.rect(x,y,sx,sy)
            
    def close_path(self,tag=None):
        """ Closes the path of the current subpath.
        
            Currently starts a new subpath -- is this what we want?
            ... Cairo starts a new subpath automatically.
        """
        self._ctx.close_path()

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
        self._ctx.curve_to(x_ctrl1, y_ctrl1, x_ctrl2, y_ctrl2, x_to, y_to)
    
#    def quad_curve_to(self, x_ctrl, y_ctrl, x_to, y_to):
#        """ Draw a quadratic bezier curve from the current point.
#
#        Parameters
#        ----------
#        x_ctrl : float
#            X-value of the control point
#        y_ctrl : float
#            Y-value of the control point.
#        x_to : float
#            X-value of the ending point of the curve
#        y_to : float
#            Y-value of the ending point of the curve.
#        """
#        # A quadratic Bezier curve is just a special case of the cubic. Reuse
#        # its implementation in case it has been implemented for the specific
#        # backend.
#        x0, y0 = self.state.current_point
#        xc1 = (x0 + x_ctrl + x_ctrl) / 3.0
#        yc1 = (y0 + y_ctrl + y_ctrl) / 3.0
#        xc2 = (x_to + x_ctrl + x_ctrl) / 3.0
#        yc2 = (y_to + y_ctrl + y_ctrl) / 3.0
#        self.curve_to(xc1, yc1, xc2, yc2, x_to, y_to)
    
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
        if cw: #not sure if I've got this the right way round
            self._ctx.arc( x, y, radius, start_angle, end_angle)
        else:
            self._ctx.arc_negative( x, y, radius, start_angle, end_angle)
    
#    def arc_to(self, x1, y1, x2, y2, radius):
#        """
#        """
#        raise NotImplementedError, "arc_to is not implemented"        
                                    
    #----------------------------------------------------------------
    # Getting infomration on paths
    #----------------------------------------------------------------

    def is_path_empty(self):
        """ Tests to see whether the current drawing path is empty
        
        What does 'empty' mean???
        """
        p = self._ctx.copy_path()
        return any(a[0] for a in p)
        
    def get_path_current_point(self):
        """ Returns the current point from the graphics context.
        
            Note:
                Currently the current_point is only affected by move_to,
                line_to, and lines.  It should also be affected by text 
                operations.  I'm not sure how rect and rects and friends
                should affect it -- will find out on Mac.
        """
        return self._ctx.get_current_point()
            
    def get_path_bounding_box(self):
        """
        cairo.Context.path_extents not yet implemented on my cairo version.
        It's in new ones though.
        
        What should this method return?
        """
        if self.is_path_empty():
            return [[0,0],[0,0]]
        p = [a[1] for a in self._ctx.copy_path()]
        p = numpy.array(p)
        return [p.min(axis=1), p.max(axis=1)]

#    def from_agg_affine(self, aff):
#        """Convert an agg.AffineTransform to a numpy matrix
#        representing the affine transform usable by kiva.affine
#        and other non-agg parts of kiva"""
#        return array([[aff[0], aff[1], 0],
#                      [aff[2], aff[3], 0],
#                      [aff[4], aff[5], 1]], float64)
        
#    def add_path(self, path):
#        """Draw a compiled path into this gc.  Note: if the CTM is
#        changed and not restored to the identity in the compiled path,
#        the CTM change will continue in this GC."""
#        # Local import to avoid a dependency if we can avoid it.
#        from enthought.kiva import agg
#
#        multi_state = 0 #For multi-element path commands we keep the previous
#        x_ctrl1 = 0     #information in these variables.
#        y_ctrl1 = 0
#        x_ctrl2 = 0
#        y_ctrl2 = 0
#        for x, y, cmd, flag in path._vertices():
#            if cmd == agg.path_cmd_line_to:
#                self.line_to(x,y)
#            elif cmd == agg.path_cmd_move_to:
#                self.move_to(x, y)
#            elif cmd == agg.path_cmd_stop:
#                self.concat_ctm(path.get_kiva_ctm())
#            elif cmd == agg.path_cmd_end_poly:
#                self.close_path()
#            elif cmd == agg.path_cmd_curve3:
#                if multi_state == 0:
#                    x_ctrl1 = x
#                    y_ctrl1 = y
#                    multi_state = 1
#                else:
#                    self.quad_curve_to(x_ctrl1, y_ctrl1, x, y)
#                    multi_state = 0
#            elif cmd == agg.path_cmd_curve4:
#                if multi_state == 0:
#                    x_ctrl1 = x
#                    y_ctrl1 = y
#                    multi_state = 1
#                elif multi_state == 1:
#                    x_ctrl2 = x
#                    y_ctrl2 = y
#                    multi_state = 2
#                elif multi_state == 2:
#                    self.curve_to(x_ctrl1, y_ctrl1, x_ctrl2, y_ctrl2, x, y)

                
                    
    #----------------------------------------------------------------
    # Clipping path manipulation
    #----------------------------------------------------------------

    def clip(self):
        """
        Should this use clip or clip_preserve
        """
        fr = self._ctx.get_fill_rule()
        self._ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
        self._ctx.clip()
        self._ctx.set_fill_rule(fr)
        
    def even_odd_clip(self):
        """
        """
        fr = self._ctx.get_fill_rule()
        self._ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        self._ctx.clip()
        self._ctx.set_fill_rule(fr)

        
    def clip_to_rect(self,x,y,width,height):
        """
            Sets the clipping path to the intersection of the current clipping 
            path with the area defined by the specified rectangle
        """
        ctx = self._ctx
        #get the current path
        p = ctx.copy_path()
        ctx.new_path()
        ctx.rectangle(x,y,width,height)
        ctx.clip()
        ctx.append_path(p)
               
#    def clip_to_rects(self):
#        """
#        """
#        pass

    def clear_clip_path(self):
        self._ctx.reset_clip()
    
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
    
    def _set_source_color(self, color):
        if len(color) == 3:
            self._ctx.set_source_rgb(*color)
        else:
            self._ctx.set_source_rgba(*color) 

    def set_fill_color(self,color):
        """
            set_fill_color takes a sequences of rgb or rgba values 
            between 0.0 and 1.0
        """
        self.state.fill_color = color
  
    def set_stroke_color(self,color):
        """
            set_stroke_color takes a sequences of rgb or rgba values 
            between 0.0 and 1.0
        """
        self.state.stroke_color = color    
    
    def set_alpha(self,alpha):
        """
        """
        self.state.alpha = alpha
                        
    #----------------------------------------------------------------
    # Drawing Images
    #----------------------------------------------------------------
        
    def draw_image(self,img,rect=None):
        """
        img is either a N*M*3 or N*M*4 numpy array, or a Kiva image
        
        Need to implement a pycairo function to create an ImageSurface 
        directly from a numpy array or a Kiva image.
        """
        pass

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
        # self.state.font = freetype.FontInfo(face_name,size,style,encoding)
        self._ctx.select_font_face(face_name, font_slant[style], font_weight[style])
        self._ctx.set_font_size(size)
        

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
        self.state.font.size = size
        
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
        return
        a,b,c,d,tx,ty = affine.affine_params(self.state.text_matrix)
        tx, ty = x,y
        self.state.text_matrix = affine.affine_from_values(a,b,c,d,tx,ty)
        # No longer uses knowledge that matrix has 3x3 representation
        #self.state.text_matrix[2,:2] = (x,y)
        
    def get_text_position(self):
        """
        """
        return
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
        self._ctx.show_text(text)

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
            
    def draw_path(self, mode=constants.FILL_STROKE):
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
        ctx = self._ctx
        fr = ctx.get_fill_rule()
        if mode in [constants.EOF_FILL, constants.EOF_FILL_STROKE]:
            ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        else:
            ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
            
        if mode in [constants.FILL, constants.EOF_FILL]:
            self._set_source_color(self.state.fill_color)
            ctx.fill()
        elif mode == constants.STROKE:
            self._set_source_color(self.state.stroke_color)
            ctx.stroke()
        elif mode in [constants.FILL_STROKE, constants.EOF_FILL_STROKE]:
            self._set_source_color(self.state.fill_color)
            ctx.fill_preserve()
            self._set_source_color(self.state.stroke_color)
            ctx.stroke()
            
        ctx.set_fill_rule(fr)

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
        
    def render_component(self, component, container_coords=False):
        """ Renders the given component.
        
        Parameters
        ----------
        component : Component
            The component to be rendered.
        container_coords : Boolean
            Whether to use coordinates of the component's container
            
        Description 
        -----------
        If *container_coords* is False, then the (0,0) coordinate of this 
        graphics context corresponds to the lower-left corner of the 
        component's **outer_bounds**. If *container_coords* is True, then the
        method draws the component as it appears inside its container, i.e., it
        treats (0,0) of the graphics context as the lower-left corner of the
        container's outer bounds.
        """
        
        x, y = component.outer_position
        w, h = component.outer_bounds
        if not container_coords:
            x = -x
            y = -y
        self.translate_ctm(x, y)
        component.draw(self, view_bounds=(0, 0, w, h))
        return
