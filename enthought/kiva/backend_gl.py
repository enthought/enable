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

import copy
import pdb

import affine       # needed for concat_ctm and get_ctm
import basecore2d   

from numpy  import alltrue, any, array, asarray, concatenate, float32, float64, int8, ones, pi, zeros
from OpenGL.GL   import *
from OpenGL.GLU  import *
#from OpenGL.GLUT import *
from constants   import *

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

DEBUG = 0

# There is no implementation of compiled paths for this backend.
CompiledPath = None

#-------------------------------------------------------------------------------
#  Helper functions:
#-------------------------------------------------------------------------------

def gcd ( n, m ):
    """Find the greatest common divisor of integers n and m.
    """
    
    if n == 0: 
       return m
    if m == 0: 
       return n
    return n * m / lcm ( n, m )

def lcm ( n, m ):
    """Return the least common multiple of integers n and m.
    """
    
    # Keep adding to an accumulator until both accumulators are the same:
    an = n
    am = m
    if (an == 0) or (am == 0): 
       return 0
    while an != am:
       if an < am:
          an += n
       else:
          am += m
    return an
    
def mycombine ( coord, vertex, weight ):
    return ( coord[0], coord[1], coord[2] )

def myvertex ( x ):
    if x:
       glVertex( x[0], x[1] )

#-------------------------------------------------------------------------------
#  'GraphicsContext' class:
#-------------------------------------------------------------------------------

class GraphicsContext ( basecore2d.GraphicsContextBase ):
    
    def __init__( self, size = ( 500,500 ), dc = None ):
        """ I don't know if we need a dc anywhere in the OpenGL version...
        """
        self._glu_tess_set = 0
        self.size          = size
        basecore2d.GraphicsContextBase.__init__( self )
                
    def begin_page ( self ):
        glClearColor( 1.0, 1.0, 1.0, 0.0 )
        glClear( GL_COLOR_BUFFER_BIT )

    #-------------------------------------------------------------------------
    # Coordinate Transform Matrix Manipulation:
    #-------------------------------------------------------------------------
    
    def get_ctm ( self ):
        """ Return the current coordinate transform matrix.  
        
            Note: untested                                   
        """ 
        glm = glGetDouble( GL_MODELVIEW_MATRIX )
        a   = glm[0,0];  d = glm[1,1]      # x, y diagonal elements
        b   = glm[1,0];  c = glm[0,1]      # x, y rotation
        tx  = glm[0,3]; ty = glm[1,3]      # x, y translation
        return affine.affine_from_values( a, b, c, d, tx, ty )

    #-------------------------------------------------------------------------
    # Handle setting pens and brushes.
    #-------------------------------------------------------------------------

    def device_update_line_state ( self ):
        """ Set the line drawing properties for the graphics context.
        """
        
        #----------------------------------------------------------------
        # Only update if the style has changed.  This and saving a copy
        # of the state (end of if block) may be more costly than just
        # going ahead and setting the line stuff every time...
        #----------------------------------------------------------------
        
        if not basecore2d.line_state_equal( self.last_drawn_line_state,
                                            self.state ):
                                    
            #----------------------------------------------------------------
            # Create a color object based on the current line color            
            #
            # I think this should be moved into each line drawing routine
            # because glColor is used to define both line and fill color.
            # This is unlikely to affect speed.
            #----------------------------------------------------------------
            
            r, g, b, a = self.state.line_color
            glColor( r, g, b, a )
            glLineWidth( self.state.line_width )
            
            #----------------------------------------------------------------
            # ignore cap and join style.  OpenGL doesn't support them as far
            # as I can tell.  The capping can probably be simulated, at least
            # for line ends, by drawing circles or squares on line ends. 
            # I don't see supporting caps on the ends of line dashes.  Also,
            # miters don't appear to have an obvious implementation.
            #----------------------------------------------------------------
            
            #----------------------------------------------------------------
            # dashing
            # not implemented
            #----------------------------------------------------------------
            
            if self.state.is_dashed():
                # line_dash is a (phase,pattern) tuple.
                # phase is ignored by OpenGL.
                factor, pattern = self.line_dash_to_gl()
                glEnable( GL_LINE_STIPPLE )
                glLineStipple( factor, pattern & 0xFFFF ) # <--- TBD: HACK!
            else:
                glDisable( GL_LINE_STIPPLE )
                                        
            #----------------------------------------------------------------
            # update the last_draw_line_state to reflect the new drawing 
            # style.  This may well be more costly than just setting all the
            # styles every time in OpenGL.
            #----------------------------------------------------------------
            
            self.last_drawn_line_state = self.state.copy()
            
    def device_update_fill_state ( self ):
        """ Set the shape filling properties for the graphics context.
            
            This only needs to be drawn before filling a path. 
            See update_line_state for comments.
            
            This uses wxBrush for wxPython.
        """
        
        #----------------------------------------------------------------
        # Test if the last color is the same as the current.  If so, we
        # don't need to mess with the brush.
        #----------------------------------------------------------------

        if any(self.last_drawn_fill_state != self.state.fill_color):
            r, g, b, a = self.state.fill_color
            glColor( r, g, b, a )
            
            #----------------------------------------------------------------
            # Update the last_draw_line_state to reflect this new pen style.
            # A copy is made so that last_drawn_fill_state is independent
            # of changes made to fill_color.
            #----------------------------------------------------------------
            
            self.last_drawn_fill_state = copy.copy( self.state.fill_color )
                        
    def device_update_font_state ( self ):
        """ 
        """
        try:
           font = self.state.font
        except AttributeError:
           self.select_font( SWISS, 18 )
           font = self.state.font
        return ### TBD
        if font.family == ROMAN:
           if font.size <= 17:
              self._glut_font = GLUT_BITMAP_TIMES_ROMAN_10
           else:
              self._glut_font = GLUT_BITMAP_TIMES_ROMAN_24
        else:  # use Helvetica
           if font.size <= 11:
              self._glut_font = GLUT_BITMAP_HELVETICA_10
           elif font.size <= 15:
              self._glut_font = GLUT_BITMAP_HELVETICA_12
           else:
              self._glut_font = GLUT_BITMAP_HELVETICA_18 
        
    def device_fill_points ( self, pts, mode ):
        """
            Needs much work for handling WINDING and ODD EVEN fill rules.
            
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
        
        if DEBUG: 
           print 'in fill'
        
        if not basecore2d.is_fully_transparent( self.state.fill_color ):          
            #print 'fill:'
            #glm = glGetInteger(GL_MODELVIEW_MATRIX)
            #print glm
            self.gl_render_points( pts, self.state.fill_color, 
                                   polygon = 1, fill = 1, mode = mode )

    def device_stroke_points ( self, pts, mode ):
        """ Draw a set of connected points using the current stroke settings.
        
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

        #--------------------------------------------------------------------
        # Only draw lines if the current settings are not fully transparent.
        # and the drawing mode is a "stroke" mode
        #-------------------------------------------------------------------- 
              
        if not basecore2d.is_fully_transparent( self.state.line_color ):

           #----------------------------------------------------------------
           # We do not try to deal with scaled line width yet:
           #----------------------------------------------------------------
           
           pass            

           #----------------------------------------------------------------
           # This if/then solves line ending problems in wxPython so that
           # the last two points are connected in the drawing.  I think
           # it may do the same here, but I'm not entirely sure.
           #----------------------------------------------------------------
           
           if alltrue( pts[0] == pts[-1] ):
              self.gl_render_points( pts, self.state.line_color, 
                                     polygon = 1, fill = 0, mode = mode ) 
           else:
              self.gl_render_points( pts, self.state.line_color, 
                                     polygon = 0, fill = 0, mode = mode )
                
    def device_draw_rect ( self, x, y, sx, sy, mode = FILL ):
        """ Not often used -- calls generally go through draw_points.
            Is mode needed?
            
            Dead code -- at least for now.
        """
        
        pts = array( ( (x,y), (x,y + sy), (x + sx, y + sy),(x + sx, y) ), float)
        self.gl_render_points( pts, self.state.fill_color, # Fill 
                               polygon = 1, fill = 1, mode = mode, convex = 1 )
        self.gl_render_points( pts, self.state. line_color, # Outline  
                               polygon = 1, fill = 0, mode = mode, convex = 1 )

    def gl_render_points ( self, pts, color, polygon = 0, fill = 0, mode = FILL,
                           convex = 0 ):
        r, g, b, a = color
        glColor( r, g, b, a )

        # Needed to correctly fill non-convex polygons:
        if polygon and fill and not convex:  
           self.gl_tesselate_polygon( pts, mode )
           return
            
        if polygon:
           poly_mode = GL_POLYGON
        else:
           poly_mode = GL_LINE_STRIP
        
        if fill:
           fill_mode = GL_FILL
        else:   
           fill_mode = GL_LINE            
        
        if a == 1.:    
           glDisable(GL_BLEND)
        else:
           glEnable( GL_BLEND )
           glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )

        glPolygonMode( GL_FRONT_AND_BACK, fill_mode )
        glVertexPointerd( pts )
        glEnableClientState( GL_VERTEX_ARRAY )
        glDrawArrays( poly_mode, 0, len( pts ) )
        glDisableClientState( GL_VERTEX_ARRAY )

        # Shouldn't be necessary:
        glDisable( GL_BLEND )

    def gl_render_points_set( self, pts_set, fill_color, stroke_color, 
                              polygon = 0, fill = 1, stroke = 1 ):        
        fr, fg, fb, fa = fill_color
        sr, sg, sb, sa = stroke_color
        pts_set        = asarray( pts_set )
        len_pts        = pts_set.shape[1]

        if polygon:
           poly_mode = GL_POLYGON
        else:
           poly_mode = GL_LINE_STRIP
        
        for pts in pts_set:    
            glVertexPointerd( pts )
            glEnableClientState( GL_VERTEX_ARRAY )
            if fill:
               glColor( fr, fa, fb, fa )
               glPolygonMode( GL_FRONT_AND_BACK, GL_FILL )
               glDrawArrays( poly_mode, 0, len_pts )
            if stroke:
               glColor( sr, sg, sb, sa )
               glPolygonMode( GL_FRONT_AND_BACK, GL_LINE )
               glDrawArrays( poly_mode, 0, len_pts )
            glDisableClientState( GL_VERTEX_ARRAY )

    def gl_tesselate_polygon ( self, pts, mode ):
        if not self._glu_tess_set:
           self.setup_glu_tess()

        if mode in [EOF_FILL, EOF_FILL_STROKE]:
           gluTessProperty( self._glu_tess, GLU_TESS_WINDING_RULE, 
                            GLU_TESS_WINDING_ODD )
        else:
           gluTessProperty( self._glu_tess, GLU_TESS_WINDING_RULE, 
                            GLU_TESS_WINDING_NONZERO )

        # Draw the points (need only a single contour)
        # This may need to go into C for speed.
        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL )        
        gluTessBeginPolygon( self._glu_tess, None )
        
        # Reported to be 10% when all points lie on x,y plane. Look here:
        # http://oss.sgi.com/cgi-bin/cvsweb.cgi/projects/ogl-sample/main/gfx/lib/glu/
        #        libtess/README?rev=1.1&content-type=text/x-cvsweb-markup        
        gluTessNormal( self._glu_tess, 0, 0, 0 )
        gluTessBeginContour( self._glu_tess )
        for pt in pts:
            thispt = array( [ pt[0], pt[1], 0 ] )
            gluTessVertex( self._glu_tess, thispt, thispt )
        gluTessEndContour( self._glu_tess )
        gluTessEndPolygon( self._glu_tess )

    def setup_glu_tess ( self ):
        self._glu_tess = gluNewTess()
        gluTessCallback( self._glu_tess, GLU_TESS_BEGIN,  glBegin )
        gluTessCallback( self._glu_tess, GLU_TESS_END,    glEnd )
        gluTessCallback( self._glu_tess, GLU_TESS_VERTEX, myvertex )
                        
        gluTessCallback( self._glu_tess, GLU_TESS_COMBINE, mycombine )
        gluTessProperty( self._glu_tess, GLU_TESS_BOUNDARY_ONLY, GL_FALSE )
        self._glu_tess_set = 1                   
        
    def device_prepare_device_ctm  (self ):
        sz = self.size
        glMatrixMode( GL_PROJECTION )
        glLoadIdentity()
        gluOrtho2D( 0.0, sz[0], 0.0, sz[1] )
        glMatrixMode( GL_MODELVIEW )
        glLoadIdentity()
        if  DEBUG: print "ctm prepared", self.gl_ctm();
                
    def device_transform_device_ctm ( self, func, args ):
        """ Default implementation for handling scaling matrices.  
        
            Many implementations will just use this function.  Others, like
            OpenGL, can benefit from overriding the method and using 
            hardware acceleration.            
        """
        if func == SCALE_CTM:
           glScalef( args[0], args[1], 1 )
        elif func == ROTATE_CTM:
           glRotate( args[0] * 180 / pi, 0, 0, 1 )        
        elif func == TRANSLATE_CTM:
           if DEBUG: print 'translate', args[0], args[1]
           glTranslate( args[0], args[1], 0 )
           if DEBUG: print 'after translate', self.gl_ctm(),'\n'
        elif func == CONCAT_CTM:
           if DEBUG: print 'concat'
           glm = self.affine_to_gl( args[0] )
           glMultMatrixf( glm )
        elif func == LOAD_CTM:
           if DEBUG: print 'load', args[0]
           glm = self.affine_to_gl( args[0] )
           glLoadMatrixf( glm )
           if DEBUG: print 'after load', self.gl_ctm(),'\n'

    def line_dash_to_gl ( self ):
        phase, pattern = self.state.line_dash
    
        # convert this to factor, gl_pattern.  This is only possible exactly if the sum of
        #  the integers in pattern is 2, 4, 8, or 16.  Otherwise, only an approximation is
        #  possible.  The approximation selected is to extend the given pattern to a multiple
        #  of 16, phase-shift the full pattern, then sample the bits to give the 16 bits
        #  available for the gl_pattern.   The bits are represented in a byte array of 1s and 0s.
        #  factor is set to the greatest common divisor of pattern and phase.  The pattern and
        #  phase are then scaled by this factor.
        L = sum( pattern )

        # Handle case where phase >= L:
        if phase >= L:
           phase = phase % L

        # Compute any factor to use:
        factor = gcd( phase, pattern[0] )
        for k in range( 1, len( pattern ) ):
            factor = gcd( factor, pattern[k] )
        if factor > 1:
           phase   /= factor
           pattern  = pattern.copy() / factor
           L       /= factor

        D    = lcm( L, 16 )
        x, n = D / L, D / 16
        
        # Make a bit-field xL long:
        bits = ones( L, 'b' )
        
        # Fill the bit-field with the given pattern:
        start = 0
        k     = 0
        for num in pattern:
            # Odd elements zero-these out in the bit-field:
            if k % 2:  
               bits[ int( start) : int( start + num ) ] = 0 # TBD: HACK
            start += num
            k     += 1

        bigbits = bits.copy()
        for k in range( 1, x ):
            bigbits = concatenate( ( bigbits, bits ) )
        if phase > 0:
           bigbits = concatenate( (bigbits[phase:], bigbits[:phase] ) )

        gl_bits = bigbits[:: int( n ) ]  # TBD: HACK take every nth sample.
    
        # Just a sanity check:
        assert(len(gl_bits)==16)   
        
        # Convert to integer (place last bit in least significant bit):
        gl_pattern = 0
        for k in range( 15, -1, -1 ):
            gl_pattern <<= 1
            gl_pattern  += gl_bits[k]

        return ( factor, gl_pattern )

    def affine_to_gl ( self, mat, load = 0 ):
        a, b, c, d, tx, ty = affine.affine_params( mat )
        glm = zeros( (4,4), float32 )
        
        # a,b,c,d,tx,ty are transposed from PDF:
        glm[0,0] = a;  glm[0,1] = b       
        glm[1,0] = c;  glm[1,1] = d       
        glm[3,0] = tx; glm[3,1] = ty      # x, y translation
        glm[2,2] = 1;  glm[3,3] = 1       # z, w diagonal elements        
        return glm

    def gl_ctm ( self ):
        return glGetFloat( GL_MODELVIEW_MATRIX )
        
    def x_device_show_text ( self, text ):
        """ Insert text at the current text position
        """
        self.device_update_font_state()
        tx, ty     = self.get_text_position()
        r, g, b, a = self.state.fill_color
        glColor3( r, g, b )
        glRasterPos( tx, ty )
        self.device_draw_rect( tx, ty, 60, 24 ) # TBD: TEMPORARY
        return ### TBD
        for char in text:
            glutBitmapCharacter( self._glut_font, ord( char ) )

    def device_draw_glyphs ( self, glyphs, tx, ty ):
        dy, dx      = glyphs.image.shape
        img         = zeros( ( dy, dx, 4 ), int8 )
        r, g, b, a  = self.state.fill_color
        img[:,:,:3] = array( ( int( 255.0 * r ), 
                               int( 255.0 * g ), 
                               int( 255.0 * b ) ), int8 )
        img[:,:,3]  = glyphs.image.astype( Int8 )[::-1]
        glRasterPos( tx + glyphs.bbox[0], ty + glyphs.bbox[1] )
        glEnable( GL_BLEND )
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )
        glDrawPixels( dx, dy, GL_RGBA, GL_UNSIGNED_BYTE, img.tostring() )
        glDisable( GL_BLEND )

    def device_set_clipping_path(self,x,y,width,height):
        glScissor( x, y, width, height )
        glEnable( GL_SCISSOR_TEST )
        return
        
    def device_destroy_clipping_path(self):
        glDisable( GL_SCISSOR_TEST )
        return    

    #------------------------------------------------------------------------
    # Overloaded
    #------------------------------------------------------------------------
    #def rect(self,x,y,sx,sy):
    #    """ Add a rectangle as a new subpath.
    #    """
    #    self._new_subpath()
    #    self.active_subpath.append( (RECT,array((x,y,sx,sy),float64)) )
            
    def flush ( self ):
        glFlush()

    def synchronize ( self ):
        # This is taking a whale of a long time (.15 sec on 1000x1000 window)
        # what gives?
        #glutSwapBuffers()
        pass


class Canvas(object):
    pass

def font_metrics_provider(*args, **kw):
    pass




