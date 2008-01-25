#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
# 
# Author: Enthought, Inc.
# Description: <Enthought kiva package component>
#------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import pdb
import time
import sys

from numpy     import array, arange, concatenate, newaxis, pi
from numpy import random
#from numpy.random import *
#from RandomArray import *
try:
    from OpenGL.GL   import *
    from OpenGL.GLU  import *
    from OpenGL.GLUT import *
except ImportError:
    raise Exception('OpenGL package needs to be installed to run this example.')

#from scipy_test.testing import *
#set_local_path('..')
#from constants   import *
#from glcore2d    import GraphicsContext
#restore_path()

from enthought.kiva.constants import *
from enthought.kiva.backend_gl import GraphicsContext

#-------------------------------------------------------------------------------
#  Utility functions:
#-------------------------------------------------------------------------------

def render_func ( gc ):
    t1 = time.clock()
    gc.begin_page()
    gc.set_stroke_color((0., 0., 1., 1.)) # blue (should be ignored)    
    gc.begin_path()
    gc.translate_ctm( 200, 200 )
    gc.rotate_ctm( pi / 8 ) 
    gc.scale_ctm( .6, .6 )    
    gc.lines( points )
    gc.set_stroke_color((0, 1., 0., 1.)) # green
    gc.stroke_path()

    gc.set_fill_color((1., 0., 0., 0.5)) # transparent red    
    gc.begin_path()
    gc.rect( 0, 0, 1, 1 )
    gc.draw_path( FILL_STROKE )
    t2 = time.clock()
    print 'render time:', t2 - t1

def calc_star ( size = 40 ):
    half_size  = size * .5
    tenth_size = size * .1
    star_pts   = [ array( ( tenth_size, 0 ) ),
                   array( ( half_size, size - tenth_size ) ),
                   array( ( size - tenth_size, 0 ) ),
                   array(( 0, half_size ) ),
                   array( ( size, half_size ) ),
                 ]
    return array( star_pts )

def add_symbol_slow ( gc, pts ):
    """ This is used to be slower by about 1/3than add_star.
    """
    gc.begin_path()
    gc.move_to( pts[0][0], pts[0][1] )
    for i in range( 5 ):
        gc.line_to( pts[i][0], pts[i][1] )
    
        # Could do this with a close path:
        gc.close_path()
        gc.draw_path( FILL_STROKE )

def add_symbol ( gc, pts ):
    gc.begin_path()
        
    # Add the star to the path
    # 1000 stars -- .15 seconds
    gc.lines( pts )
    gc.close_path()
    
    # 1000 stars - .20 seconds
    gc.draw_path( FILL_STROKE )

#-------------------------------------------------------------------------------
#  'StarSample' class:
#-------------------------------------------------------------------------------

class StarSampler:

    def __init__ ( self, star_count = 20, scale = 1.0, rotate = 0.0, 
                         line_width = 3 ):
        self.star_count = star_count
        symbol_pts      = calc_star()
        
        # Connect the last point to the first on the star.
        # PDF doesn't draw it correctly if I do this.
        # now we call close_path in the add_symbol funciton
        #self.symbol_pts = concatenate((symbol_pts,symbol_pts[:1,:]))
        self.symbol_pts = symbol_pts
        random.seed(10000)
        self.star_pos   = random.randint( 20, 300, ( star_count, 2 ) )
        self.scale      = scale
        self.rotate     = rotate
        self.line_width = line_width
        
    def size ( self ):
        """ This really needs to be calculated, but we'll hard code for now.
        """
        return ( 170 * self.scale, 250 * self.scale )
    
    def scale ( self, scale ):
        self.scale = scale
            
    def draw ( self, gc ):
        gc.save_state()
        # The rotation isn't handled correctly on most platforms.
        
        # fill stars with red
        gc.set_fill_color((0, 1, 0, 1))
        
        # Outline stars with a wide black line:
        gc.set_stroke_color((0, 0, 0, 1))
        gc.set_line_width( self.line_width )

        for x, y in self.star_pos:
            gc.save_state()
            gc.translate_ctm( x, y ) 
            gc.rotate_ctm( self.rotate )
            gc.scale_ctm( self.scale, self.scale )
            add_symbol( gc, self.symbol_pts )
            gc.restore_state()
            
        gc.set_fill_color((0, 0, 0, 1))
        gc.set_font_size( 24 )
        gc.set_text_position( 20, 180 )
        gc.show_text( "Stars, count=%d" % self.star_count )
        gc.set_text_position( 20, 150 )
        gc.show_text( "scale=%2.1f" %  self.scale )        
        gc.restore_state()

#-----------------------------------------------------------------------------
# NoiseSampler:
#-----------------------------------------------------------------------------

def render_func2 ( gc = None ):
    global zoom, rotate
    if gc is None:
       gc = GraphicsContext()
    t1 = time.clock()
    gc.begin_page()
#    gc.translate_ctm( 200, 200 )
    gc.scale_ctm(zoom, zoom)
    s = StarSampler( star_count = 30, scale = 2.0, rotate = angle )
    s.draw( gc )    
    t2 = time.clock()
    print 'render time:', t2 - t1

def render_func3 ( gc = None ):
    if gc is None:
       gc = GraphicsContext()
    global angle
    stars = 30
    print "calling star sampler with angle", angle
    s = StarSampler( star_count = stars, rotate = angle)
    s.draw(gc)
    
def display ( *args ):
    #print "display"
    clear_buffer()

    # Import profile:
    gc = GraphicsContext()
    
    render_func3(gc)
    t1 = time.clock()
    gc.flush()
    gc.synchronize()
    t2 = time.clock()
    glutSwapBuffers()
        
def halt ( ):
    pass

def keyboard (key, x, y):
    global angle, zoom
    if key == 'q':
        sys.exit()
    elif key in ("+", "="):
        angle += delta_angle
    elif key == "-":
        angle -= delta_angle
    elif key == "z":
        zoom += zoom_fac
    elif key == "Z":
        zoom -= zoom_fac
    display()

def mouse ( button, state, x, y ):
    global angle, delta_angle, move_x, move_y, move_length, halted
    if button == GLUT_LEFT_BUTTON:
       angle = angle + delta_angle
       display()
    elif button == GLUT_RIGHT_BUTTON:
       angle = angle - delta_angle
       display()
    elif button == GLUT_MIDDLE_BUTTON and state == GLUT_DOWN:
       if halted:
          glutIdleFunc( display )
          halted = 0
       else:
          glutIdleFunc( halt )
          halted = 1
    #move_x = move_length * cos(angle)
    #move_y = move_length * sin(angle)

def setup_viewport ( ):
    glMatrixMode( GL_PROJECTION )
    glLoadIdentity()
    glOrtho( 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 )
    glMatrixMode( GL_MODELVIEW )
    
def reshape ( w, h ):
    glViewport( 0, 0, w, h )
    setup_viewport()

def clear_buffer ( ):
    glClearColor( 1.0, 1.0, 1.0, 0.0 )
    glClear( GL_COLOR_BUFFER_BIT )

def main ( ):
    glutInit( sys.argv )
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA | GLUT_ALPHA )
    glutInitWindowSize( 500, 500 )
    glutCreateWindow( 'Stars' )
    glutShowWindow()
    setup_viewport()
    glutReshapeFunc( reshape )
    glutDisplayFunc( display )
    glutIdleFunc( None )
    glutMouseFunc( mouse )
    glutKeyboardFunc( keyboard )
    clear_buffer()
    glutMainLoop()

if __name__ == "__main__":
    halted      = 0
    N           = 30000
    x           = arange( N ) * 1. / N * 1000.
    y           = random.random( N ) * 1000.
    points      = concatenate( ( x[ :, newaxis ], y[ :, newaxis ] ), axis = -1 )
    star_points = calc_star()
    angle       = 0.0
    delta_angle = 25.0
    zoom = 1.0
    zoom_fac = 0.2

    main()
