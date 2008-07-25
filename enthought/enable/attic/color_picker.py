#-------------------------------------------------------------------------------
#
#  Defines an Enable-compatible color picker component.
#
#  Written by: David C. Morrill
#
#  Date: 01/16/2005
#
#  Symbols defined: ColorPicker
#
#  (c) Copyright 2005 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

# Major library imports
import numpy
from numpy import zeros, ones, floor, putmask, uint8, choose, concatenate, \
                  repeat, newaxis, arange

# Enthought library imports
from enthought.enable.colors import ColorTrait
from enthought.kiva.traits.kiva_font_trait import KivaFont
from enthought.traits.api import Enum, Event, Str, true

# Local, relative imports
from base import coordinates_to_bounds, HCENTER, LEFT, GraphicsContextArray
from colors import color_table
from component import Component


#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

black_color = color_table["black"]
white_color = color_table["white"]
cursor_color = ( white_color, black_color )

#-------------------------------------------------------------------------------
#  'ColorPicker' class:
#-------------------------------------------------------------------------------

class ColorPicker ( Component ):

    #---------------------------------------------------------------------------
    #  Geometry information for the various editor styles:
    #---------------------------------------------------------------------------

    simple_style = (
        (   4, 4,  66, 20,  0,  0 ), # 'Color well' bounds
        (  72, 5, 136, 19,  3, 16 ), # 'Red'   slider bounds (and slider size)
        ( 140, 5, 204, 19,  3, 16 ), # 'Green' slider bounds (and slider size)
        ( 208, 5, 272, 19,  3, 16 ), # 'Blue'  slider bounds (and slider size)
        ( 276, 5, 340, 19,  3, 16 )  # 'Alpha' slider bounds (and slider size)
    )

    custom_style_rgb = (
        (   8,  8,  81, 81,  0,  0 ), # 'Color well' bounds
        ( 105, 60, 232, 83,  3, 25 ), # 'Red'   slider bounds (and slider size)
        ( 105, 33, 232, 56,  3, 25 ), # 'Green' slider bounds (and slider size)
        ( 105,  6, 232, 29,  3, 25 ), # 'Blue'  slider bounds (and slider size)
        (  87,  6, 101, 82, 14,  3 ), # 'Alpha' slider bounds (and slider size)
        ( 237, 46, 254, 64, 'hsv'  ), # 'H' button bounds
        ( 237, 26, 254, 44, 'hsv2' ), # 'S' button bounds
        ( 237,  6, 254, 24, 'hsv3' )  # 'V' button bounds
    )

    custom_style_hsv = (
        (   8,  8,  81, 81,  0,  0 ), # 'Color well' bounds
        ( 216,  6, 234, 83, 20,  3 ), # 'H' slider bounds (and slider size)
        ( 105,  6, 212, 83,  3, 25 ), # 'SV' slider bounds
        (  87,  6, 101, 82, 14,  3 ), # 'Alpha' slider bounds (and slider size)
        ( 237, 66, 254, 84, 'rgb'  ), # 'R' button bounds
        ( 237, 26, 254, 44, 'hsv2' ), # 'S' button bounds
        ( 237,  6, 254, 24, 'hsv3' )  # 'V' button bounds
    )

    custom_style_hsv2 = (
        (   8,  8,  81, 81,  0,  0 ), # 'Color well' bounds
        ( 216,  6, 234, 83, 20,  3 ), # 'S' slider bounds (and slider size)
        ( 105,  6, 212, 83,  3, 25 ), # 'HV' slider bounds
        (  87,  6, 101, 82, 14,  3 ), # 'Alpha' slider bounds (and slider size)
        ( 237, 66, 254, 84, 'rgb'  ), # 'R' button bounds
        ( 237, 46, 254, 64, 'hsv'  ), # 'H' button bounds
        ( 237,  6, 254, 24, 'hsv3' )  # 'V' button bounds
    )

    custom_style_hsv3 = (
        (   8,  8,  81, 81,  0,  0 ), # 'Color well' bounds
        ( 216,  6, 234, 83, 20,  3 ), # 'V' slider bounds (and slider size)
        ( 105,  6, 212, 83,  3, 25 ), # 'HS' slider bounds
        (  87,  6, 101, 82, 14,  3 ), # 'Alpha' slider bounds (and slider size)
        ( 237, 66, 254, 84, 'rgb'  ), # 'R' button bounds
        ( 237, 46, 254, 64, 'hsv'  ), # 'H' button bounds
        ( 237, 26, 254, 44, 'hsv2' )  # 'S' button bounds
    )

    simple_style_na = (
        (   4, 4,  66, 20,  0,  0 ), # 'Color well' bounds
        (  72, 5, 136, 19,  3, 16 ), # 'Red'   slider bounds (and slider size)
        ( 140, 5, 204, 19,  3, 16 ), # 'Green' slider bounds (and slider size)
        ( 208, 5, 272, 19,  3, 16 )  # 'Blue'  slider bounds (and slider size)
    )

    custom_style_rgb_na = (
        (   8,  8,  81, 81,  0,  0 ), # 'Color well' bounds
        (  88, 60, 215, 83,  3, 25 ), # 'Red'   slider bounds (and slider size)
        (  88, 33, 215, 56,  3, 25 ), # 'Green' slider bounds (and slider size)
        (  88,  6, 215, 29,  3, 25 ), # 'Blue'  slider bounds (and slider size)
        ( 220, 46, 237, 64, 'hsv'  ), # 'H' button bounds
        ( 220, 26, 237, 44, 'hsv2' ), # 'S' button bounds
        ( 220,  6, 237, 24, 'hsv3' )  # 'V' button bounds
    )

    custom_style_hsv_na = (
        (   8,  8,  81, 81,  0,  0 ), # 'Color well' bounds
        ( 199,  6, 217, 83, 20,  3 ), # 'H' slider bounds (and slider size)
        (  88,  6, 195, 83,  3, 25 ), # 'SV' slider bounds
        ( 220, 66, 237, 84, 'rgb'  ), # 'R' button bounds
        ( 220, 26, 237, 44, 'hsv2' ), # 'S' button bounds
        ( 220,  6, 237, 24, 'hsv3' )  # 'V' button bounds
    )

    custom_style_hsv2_na = (
        (   8,  8,  81, 81,  0,  0 ), # 'Color well' bounds
        ( 199,  6, 217, 83, 20,  3 ), # 'S' slider bounds (and slider size)
        (  88,  6, 195, 83,  3, 25 ), # 'HV' slider bounds
        ( 220, 66, 237, 84, 'rgb'  ), # 'R' button bounds
        ( 220, 46, 237, 64, 'hsv'  ), # 'H' button bounds
        ( 220,  6, 237, 24, 'hsv3' )  # 'V' button bounds
    )

    custom_style_hsv3_na = (
        (   8,  8,  81, 81,  0,  0 ), # 'Color well' bounds
        ( 199,  6, 217, 83, 20,  3 ), # 'V' slider bounds (and slider size)
        (  88,  6, 195, 83,  3, 25 ), # 'HS' slider bounds
        ( 220, 66, 237, 84, 'rgb'  ), # 'R' button bounds
        ( 220, 46, 237, 64, 'hsv'  ), # 'H' button bounds
        ( 220, 26, 237, 44, 'hsv2' )  # 'S' button bounds
    )

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    color      = ColorTrait                    # Color being edited
    bg_color   = ColorTrait( 'white' )         # Component background color
    style      = Enum( 'simple', 'custom' )   # Editor style
    mode       = Enum( 'rgb', 'hsv', 'hsv2', 'hsv3' ) # Color space mode
    edit_alpha = true                         # Should alpha channel be edited?
    text       = Str( '%R' )                  # Text to display in color well
    font       = KivaFont( 'modern 8' )        # Font to use to display text
    clicked    = Event                        # Fired on color well clicked
    auto_set   = true                         # Set color while dragging

    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, **traits ):
        super( ColorPicker, self ).__init__( **traits )
        self._style_changed()
        self.event_state = self.mode

    #---------------------------------------------------------------------------
    #  Handles the edit color being changed:
    #---------------------------------------------------------------------------

    def _color_changed ( self ):
        self._hsv_color = rgb_to_hsv( *self.color_ )
        self.redraw()

    #---------------------------------------------------------------------------
    #  Handles the editing style being changed:
    #---------------------------------------------------------------------------

    def _style_changed ( self ):
        alpha = ''
        if not self.edit_alpha:
            alpha = '_na'
        if self.style == 'simple':
            self._style_info = getattr( self, 'simple_style' + alpha )
            size             = ( 346, 24 )
            self.event_state = 'rgb'
            self._image      = self.image_for( '=color_picker_simple' + alpha )
            if alpha != '':
                size = ( 278, 24 )
        else:
            self.event_state = mode = self.mode
            self._style_info = getattr( self, 'custom_style_' + mode + alpha )
            size             = ( 259, 89 )
            self._image      = self.image_for( '=color_picker_custom_' +
                                               mode + alpha )
            if alpha != '':
                size = ( 242, 89 )
        self.min_width  = self.max_width  = size[0]
        self.min_height = self.max_height = size[1]
        self.redraw()

    #---------------------------------------------------------------------------
    #  Handles the editing mode being changed:
    #---------------------------------------------------------------------------

    def _mode_changed ( self ):
        self._style_changed()

    #---------------------------------------------------------------------------
    #  Handles the edit alpha state being changed:
    #---------------------------------------------------------------------------

    def _edit_alpha_changed ( self ):
        self._style_changed()

    #---------------------------------------------------------------------------
    #  Handles a redraw of the component:
    #---------------------------------------------------------------------------

    def _draw ( self, gc, view_bounds, mode ):
        gc.save_state()
        x, y, dx, dy = self.bounds

        # Fill the component with the background color:
        gc.set_fill_color( self.bg_color_ )
        gc.begin_path()
        gc.rect( x, y, dx, dy )
        gc.fill_path()

        # Draw the color picker image:
        image = self._image
        gc.draw_image( image, ( x, y, image.width(), image.height() ) )

        # Fill in the 'color well' with the current color:
        sx, sy, ex, ey, idx, idy = self._style_info[0]
        gc.set_fill_color( self.color_ )
        gc.begin_path()
        gc.rect( x + sx, y + sy, ex - sx, ey - sy )
        gc.fill_path()

        if self.style == 'simple':
            self._draw_rgb( gc )
        else:
            getattr( self, '_draw_' + self.mode )( gc )

        if self.text != '':
            self._draw_text( gc )

        gc.restore_state()

    #---------------------------------------------------------------------------
    #  Draws the RGB color space slider bars:
    #---------------------------------------------------------------------------

    def _draw_rgb ( self, gc ):
        si = self._style_info
        sx, sy, ex, ey, bdx, dy = si[1]

        # Compute the size of each slider bar:
        dx, dy = (ex - sx), (ey - sy)

        # Compute the gradient information:
        gradient = arange( 255.0, -0.001, -255.0 / (dx - 1) )

        # Allocate a temporary graphics context to put the gradient in:
        #tgc = self.window._create_gc( ( dx, dy ) )
        tgc = GraphicsContextArray((dx, dy))

        # Draw the 'red' slider bar:
        self._draw_slider_bar( gc, tgc,
                    rgb_gradient( gradient, 0.0, 0.0, dx, dy ), si[1], dx, dy )

        # Draw the 'green' slider bar:
        self._draw_slider_bar( gc, tgc,
                    rgb_gradient( 0.0, gradient, 0.0, dx, dy ), si[2], dx, dy )

        # Draw the 'blue' slider bar:
        self._draw_slider_bar( gc, tgc,
                    rgb_gradient( 0.0, 0.0, gradient, dx, dy ), si[3], dx, dy )

        # Draw the sliders for the current color:
        color = self.color_
        for i in range( ( 3, 4 )[ self.edit_alpha ] ):
            self._draw_slider( gc, color[i], si[i+1] )

    #---------------------------------------------------------------------------
    #  Draws the HSV color space slider bars:
    #---------------------------------------------------------------------------

    def _draw_hsv ( self, gc ):
        # Get the HSV values for the current color:
        h, s, v = self._hsv_color

        si = self._style_info

        # Compute the size of the 'H' slider bar:
        sx, sy, ex, ey, bdx, bdy = si[1]
        dx, dy = (ex - sx), (ey - sy)

        # Draw the 'H' slider bar and slider:
        gradient = hsv_to_rgb( arange( 0.0, 360.0, 360.0 / dy )[:, newaxis ],
                               1.0, 1.0, dx, dy )
        self._draw_slider_bar( gc, None, gradient, si[1], dx, dy )
        self._draw_slider( gc, 1.0 - (h / 360.0), si[1] )

        # Compute the size of the 'SV' slider bar:
        sx, sy, ex, ey, bdx, bdy = si[2]
        dx, dy = (ex - sx), (ey - sy)

        gradient = hsv_to_rgb( h,
                       arange( 1.0, -0.0001, -1.0 / (dy - 1) )[:, newaxis],
                       arange( 1.0, -0.0001, -1.0 / (dx - 1) ),
                       dx, dy )
        self._draw_slider_bar( gc, None, gradient, si[2], dx, dy )

        # Draw the 'SV' cursor:
        x = int( sx + (1.0 - v) * dx ) + 0.5
        y = int( sy + s * dy ) + 0.5
        self._draw_cursor( gc, x, y, si[2][0:4],
                           cursor_color[ (v >= 0.6) and (s <= 0.5) ] )

        # Draw the 'A' slider for the current color:
        if self.edit_alpha:
            self._draw_slider( gc, self.color_[3], si[3] )

    #---------------------------------------------------------------------------
    #  Draws the HSV2 color space slider bars:
    #---------------------------------------------------------------------------

    def _draw_hsv2 ( self, gc ):
        # Get the HSV values for the current color:
        h, s, v = self._hsv_color

        si = self._style_info

        # Compute the size of the 'S' slider bar:
        sx, sy, ex, ey, bdx, bdy = si[1]
        dx, dy = (ex - sx), (ey - sy)

        # Draw the 'S' slider bar and slider:
        gradient = hsv_to_rgb(
            h, arange( 0.0, 1.0001, 1.0 / (dy - 1) )[:, newaxis ], v, dx, dy )
        self._draw_slider_bar( gc, None, gradient, si[1], dx, dy )
        self._draw_slider( gc, 1.0 - s, si[1] )

        # Compute the size of the 'HV' slider bar:
        sx, sy, ex, ey, bdx, bdy = si[2]
        dx, dy = (ex - sx), (ey - sy)

        # Draw the 'HV' slider bar:
        gradient = hsv_to_rgb( arange( 0.0, 360.0, 360.0 / dx ), 1.0,
                       arange( 1.0, -0.0001, -1.0 / (dy - 1) )[:, newaxis],
                       dx, dy )
        self._draw_slider_bar( gc, None, gradient, si[2], dx, dy )

        # Draw the 'HV' cursor:
        x = int( sx + (h / 360.0) * dx ) + 0.5
        y = int( sy + v * dy ) + 0.5
        self._draw_cursor( gc, x, y, si[2][0:4],
                           cursor_color[ (v > 0.7) and (h < 190.0) ] )

        # Draw the 'A' slider for the current color:
        if self.edit_alpha:
            self._draw_slider( gc, self.color_[3], si[3] )

    #---------------------------------------------------------------------------
    #  Draws the HSV3 color space slider bars:
    #---------------------------------------------------------------------------

    def _draw_hsv3 ( self, gc ):
        # Get the HSV values for the current color:
        h, s, v = self._hsv_color

        si = self._style_info

        # Compute the size of the 'V' slider bar:
        sx, sy, ex, ey, bdx, bdy = si[1]
        dx, dy = (ex - sx), (ey - sy)

        # Draw the 'V' slider bar and slider:
        gradient = hsv_to_rgb(
            h, s, arange( 0.0, 1.0001, 1.0 / (dy - 1) )[:, newaxis ], dx, dy )
        self._draw_slider_bar( gc, None, gradient, si[1], dx, dy )
        self._draw_slider( gc, 1.0 - v, si[1] )

        # Compute the size of the 'HS' slider bar:
        sx, sy, ex, ey, bdx, bdy = si[2]
        dx, dy = (ex - sx), (ey - sy)

        # Draw the 'HS' slider bar:
        gradient = hsv_to_rgb( arange( 0.0, 360.0, 360.0 / dx ),
                       arange( 1.0, -0.0001, -1.0 / (dy - 1) )[:, newaxis],
                       1.0, dx, dy )
        self._draw_slider_bar( gc, None, gradient, si[2], dx, dy )

        # Draw the 'HS' cursor:
        x = int( sx + (h / 360.0) * dx ) + 0.5
        y = int( sy + s * dy ) + 0.5
        self._draw_cursor( gc, x, y, si[2][0:4],
                           cursor_color[ (s < 0.5) or (h < 190.0) ] )

        # Draw the 'A' slider for the current color:
        if self.edit_alpha:
            self._draw_slider( gc, self.color_[3], si[3] )

    #---------------------------------------------------------------------------
    #  Draws a 2D cursor at a specified (x,y) location:
    #---------------------------------------------------------------------------

    def _draw_cursor ( self, gc, x, y, bounds, color ):
        gc.set_stroke_color( color )
        gc.save_state()
        gc.clip_to_rect(*bounds)
        gc.begin_path()
        gc.move_to( x - 6, y );   gc.line_to( x - 3, y )
        gc.move_to( x + 3, y );   gc.line_to( x + 6, y )
        gc.move_to( x, y - 6 );   gc.line_to( x, y - 3 )
        gc.move_to( x, y + 3 );   gc.line_to( x, y + 6 )
        gc.stroke_path()
        gc.restore_state()

    #---------------------------------------------------------------------------
    #  Draws a slider bar using a specified gradient:
    #---------------------------------------------------------------------------

    def _draw_slider_bar ( self, gc, tgc, gradient, bi, dx, dy ):
        if tgc is None:
            tgc = GraphicsContextArray((dx, dy))
        tgc.bmp_array[:,:,:] = gradient
        gc.draw_image( tgc, ( bi[0], bi[1], dx, dy ) )

    #---------------------------------------------------------------------------
    #  Draws a slider bar:
    #---------------------------------------------------------------------------

    def _draw_slider ( self, gc, color, bounds ):
        x, y, dx, dy = self.bounds
        sx, sy, ex, ey, bdx, bdy = bounds
        cdx = ex - sx
        cdy = ey - sy
        gc.set_stroke_color( black_color )
        gc.set_line_width( 3 )
        gc.begin_path()
        if cdx > cdy:
            dx = x + sx + int( (1.0 - color) * cdx ) + 0.5
            dy = y + sy + (cdy - bdy) / 2
            gc.move_to( dx, dy )
            gc.line_to( dx, dy + bdy )
            gc.stroke_path()
            gc.set_stroke_color( white_color )
            gc.set_line_width( 1 )
            gc.begin_path()
            gc.move_to( dx, dy + 1 )
            gc.line_to( dx, dy + bdy - 1 )
        else:
            dx = x + sx + (cdx - bdx) / 2
            dy = y + sy + int( color * cdy ) + 0.5
            gc.move_to( dx, dy )
            gc.line_to( dx + bdx, dy )
            gc.stroke_path()
            gc.set_stroke_color( white_color )
            gc.set_line_width( 1 )
            gc.begin_path()
            gc.move_to( dx + 1, dy )
            gc.line_to( dx + bdx - 1, dy )
        gc.stroke_path()

    #---------------------------------------------------------------------------
    #  Draws the formatted text into the color well:
    #---------------------------------------------------------------------------

    def _draw_text ( self, gc ):
        h, s, v    = self._hsv_color
        r, g, b, a = self.color_
        sr         = int( 255 * r )
        sg         = int( 255 * g )
        sb         = int( 255 * b )
        sa         = int( 255 * a )
        text       = self.text
        if text == '%x':
            text = '0x%02x%02x%02x%02x' % ( 255 - sa, sr, sg, sb )
        elif text == '%X':
            text = '0x%02X%02X%02X%02X' % ( 255 - sa, sr, sg, sb )
        elif text == '%w':
            text = '#%02x%02x%02x' % ( sr, sg, sb )
        elif text == '%W':
            text = '#%02X%02X%02X' % ( sr, sg, sb )
        elif text == '%r':
            text = 'R = %.3f\nG = %.3f\nB = %.3f' % ( r, g, b )
        elif text == '%R':
            text = 'R = %03d\nG = %03d\nB = %03d' % ( sr, sg, sb )
        elif text == '%a':
            text = 'R = %.3f\nG = %.3f\nB = %.3f\nA = %.3f' % ( r, g, b, a )
        elif text == '%A':
            text = 'R = %03d\nG = %03d\nB = %03d\nA = %03d' % ( sr, sg, sb, sa )
        lines      = text.split( '\n' )
        n          = len( lines )
        bounds     = list( coordinates_to_bounds( self._style_info[0][0:4] ) )
        bounds[3]  = dy = (bounds[3] / n)
        bounds[1] += (dy * (n - 1))
        color      = cursor_color[ ((v > 0.7) and (h < 190.0)) or (a < 0.4) ]
        alignment  = HCENTER
        if n > 1:
            alignment  = LEFT
            bounds[0] += 4
            bounds[2] -= 4
        for text in lines:
            gc.text( text, bounds,
                     font      = self.font,
                     color     = color,
                     y_offset  = 0.5,
                     alignment = alignment )
            bounds[1] -= dy

    #---------------------------------------------------------------------------
    #  Mouse event state handling methods:
    #---------------------------------------------------------------------------

    #-- 'rgb' state: --------------------------------------------------------

    def rgb_left_down ( self, event ):
        x = event.x
        y = event.y
        for i, bounds in enumerate( self._style_info ):
            bx, by, idx, idy = self.bounds
            sx, sy, ex, ey   = bounds[0:4]
            if ((bx + sx) <= x < (bx + ex)) and ((by + sy) <= y < (by + ey)):
                event.window.mouse_owner = self
                self._index = i
                if isinstance(bounds[-1], basestring):
                    self.event_state = 'button_pending'
                elif i == 0:
                    self.event_state = 'click_pending'
                else:
                    self.event_state += '_sliding'
                    getattr( self, self.event_state + '_mouse_move' )( event )

    #-- 'rgb sliding' state: -------------------------------------------------------

    def rgb_sliding_mouse_move ( self, event ):
        color                    = list( self.color_ )
        index                    = self._index - 1
        bx, by, idx, idy         = self.bounds
        sx, sy, ex, ey, idx, idy = self._style_info[ index + 1 ]
        x, y = event.x, event.y
        x   -= bx
        y   -= by
        dx   = ex - sx
        dy   = ey - sy
        if dx > dy:
            x = max( sx, min( x, ex ) )
            color[ index ] = 1.0 - (float( x - sx ) / dx)
        else:
            y = max( sy, min( y, ey ) )
            color[ index ] = float( y - sy ) / dy
        if self.auto_set:
            self.color  = tuple( color )
        else:
            self.color_     = tuple( color )
            self._hsv_color = rgb_to_hsv( *color )
            self.redraw()

    def rgb_sliding_left_up ( self, event ):
        event.window.mouse_owner  = None
        self.event_state         = 'rgb'
        self.color               = self.color_

    #-- 'hsv' state: --------------------------------------------------------

    def hsv_left_down ( self, event ):
        self.rgb_left_down( event )

    #-- 'hsv sliding' state: -------------------------------------------------------

    def hsv_sliding_mouse_move ( self, event ):
        color            = list( self.color_ )
        index            = self._index
        bx, by, idx, idy = self.bounds
        sx, sy, ex, ey   = self._style_info[ index ][0:4]
        x, y = event.x, event.y
        x   -= bx
        y   -= by
        dx   = ex - sx
        dy   = ey - sy
        h, s, v = self._hsv_color
        if index == 3:
            y = max( sy, min( y, ey ) )
            color[3] = float( y - sy ) / dy
        else:
            if index == 1:
                y = max( sy + 1, min( y, ey - 1 ) )
                h = (1.0 - (float( y - sy ) / dy)) * 360.0
            else:
                x = max( sx, min( x, ex ) )
                v = 1.0 - (float( x - sx ) / dx)
                y = max( sy, min( y, ey ) )
                s = float( y - sy ) / dy
            b, g, r, a = hsv_to_rgb( h, s, v, 1, 1 )[0,0,:]
            color = ( r / 255.0, g / 255.0, b / 255.0, color[3] )
        if self.auto_set:
            self.color  = tuple( color )
        else:
            self.color_ = tuple( color )
            self.redraw()
        self._hsv_color = ( h, s, v )

    def hsv_sliding_left_up ( self, event ):
        event.window.mouse_owner  = None
        self.event_state         = self.event_state[
                                        0: self.event_state.find( '_' ) ]
        hsv                      = self._hsv_color
        self.color               = self.color_
        self._hsv_color          = hsv

    #-- 'hsv2' state: --------------------------------------------------------

    def hsv2_left_down ( self, event ):
        self.rgb_left_down( event )

    #-- 'hsv2 sliding' state: -------------------------------------------------------

    def hsv2_sliding_mouse_move ( self, event ):
        color            = list( self.color_ )
        index            = self._index
        bx, by, idx, idy = self.bounds
        sx, sy, ex, ey   = self._style_info[ index ][0:4]
        x, y = event.x, event.y
        x   -= bx
        y   -= by
        dx   = ex - sx
        dy   = ey - sy
        h, s, v = self._hsv_color
        if index == 3:
            y = max( sy, min( y, ey ) )
            color[3] = float( y - sy ) / dy
        else:
            if index == 1:
                y = max( sy, min( y, ey ) )
                s = 1.0 - (float( y - sy ) / dy)
            else:
                x = max( sx, min( x, ex - 1 ) )
                h = (float( x - sx ) / dx) * 360.0
                y = max( sy, min( y, ey ) )
                v = float( y - sy ) / dy
            b, g, r, a = hsv_to_rgb( h, s, v, 1, 1 )[0,0,:]
            color = ( r / 255.0, g / 255.0, b / 255.0, color[3] )
        if self.auto_set:
            self.color  = tuple( color )
        else:
            self.color_ = tuple( color )
            self.redraw()
        self._hsv_color = ( h, s, v )

    def hsv2_sliding_left_up ( self, event ):
        self.hsv_sliding_left_up ( event )

    #-- 'hsv3' state: --------------------------------------------------------

    def hsv3_left_down ( self, event ):
        self.rgb_left_down( event )

    #-- 'hsv3 sliding' state: -------------------------------------------------------

    def hsv3_sliding_mouse_move ( self, event ):
        color            = list( self.color_ )
        index            = self._index
        bx, by, idx, idy = self.bounds
        sx, sy, ex, ey   = self._style_info[ index ][0:4]
        x, y = event.x, event.y
        x   -= bx
        y   -= by
        dx   = ex - sx
        dy   = ey - sy
        h, s, v = self._hsv_color
        if index == 3:
            y = max( sy, min( y, ey ) )
            color[3] = float( y - sy ) / dy
        else:
            if index == 1:
                y = max( sy, min( y, ey ) )
                v = 1.0 - (float( y - sy ) / dy)
            else:
                x = max( sx, min( x, ex - 1 ) )
                h = (float( x - sx ) / dx) * 360.0
                y = max( sy, min( y, ey ) )
                s = float( y - sy ) / dy
            b, g, r, a = hsv_to_rgb( h, s, v, 1, 1 )[0,0,:]
            color = ( r / 255.0, g / 255.0, b / 255.0, color[3] )
        if self.auto_set:
            self.color  = tuple( color )
        else:
            self.color_ = tuple( color )
            self.redraw()
        self._hsv_color = ( h, s, v )

    def hsv3_sliding_left_up ( self, event ):
        self.hsv_sliding_left_up ( event )

    #-- 'click pending' state: -------------------------------------------------

    def click_pending_left_up ( self, event ):
        event.window.mouse_owner = None
        if self.style == 'simple':
            self.event_state = 'rgb'
        else:
            self.event_state = self.mode
        bx, by, idx, idy         = self.bounds
        sx, sy, ex, ey, idx, idy = self._style_info[0]
        if (((bx + sx) <= event.x < (bx + ex)) and
            ((by + sy) <= event.y < (by + ey))):
            self.clicked = True

    #-- 'button pending' state: ------------------------------------------------

    def button_pending_left_up ( self, event ):
        event.window.mouse_owner = None
        self.event_state        = self.mode
        bx, by, idx, idy        = self.bounds
        bounds                  = self._style_info[ self._index ]
        sx, sy, ex, ey          = bounds[0:4]
        if (((bx + sx) <= event.x < (bx + ex)) and
            ((by + sy) <= event.y < (by + ey))):
            self.mode = bounds[4]

#-------------------------------------------------------------------------------
#  Creates an RGB color gradient:
#-------------------------------------------------------------------------------

def rgb_gradient ( r, g, b, width, height ):
    s  = ones( ( width, ), float )
    rs = r * s
    gs = g * s
    bs = b * s
    a  = 255.0 * s
    return repeat(
               concatenate( ( bs[ newaxis, :, newaxis ],
                              gs[ newaxis, :, newaxis ],
                              rs[ newaxis, :, newaxis ],
                              a[  newaxis, :, newaxis ] ), axis = -1
               ).astype( uint8 ), height )

#-------------------------------------------------------------------------------
#  Converts a HSV color value to an RGB color value array of a specified size:
#-------------------------------------------------------------------------------

def hsv_to_rgb ( h, s, v, width, height ):
    h  = as_2d_array(     h, width, height )
    s  = as_2d_array(     s, width, height )
    v  = as_2d_array(     v, width, height )
    a  = as_2d_array( 255.0, width, height )
    h  = h / 60.0
    i  = floor( h )
    f  = h - i
    p  = v * (1.0 - s)
    q  = v * (1.0 - (s * f))
    t  = v * (1.0 - (s * (1.0 - f)))
    c  = i.astype( numpy.int )
    putmask( c, (s == 0.0), 6 )
    r  = 255.0 * choose( c, ( v, q, p, p, t, v, v ) )
    g  = 255.0 * choose( c, ( t, v, v, q, p, p, v ) )
    b  = 255.0 * choose( c, ( p, p, t, v, v, q, v ) )
    return concatenate( ( b[ :, :, newaxis ],
                          g[ :, :, newaxis ],
                          r[ :, :, newaxis ],
                          a[ :, :, newaxis ] ), axis = -1
           ).astype( uint8 )

#-------------------------------------------------------------------------------
#  Converts a scalar, row or column vector to a 2D array of the specified size:
#-------------------------------------------------------------------------------

def as_2d_array ( value, width, height ):
    if type( value ) is numpy.float:
        return value * ones( ( height, width ) )
    s = value.shape
    if len( s ) == 1:
        return repeat( value[ newaxis, : ], height )
    if s[0] == 1:
        return repeat( value, height )
    return repeat( value, width, axis = -1 )

#-------------------------------------------------------------------------------
#  Converts an RGB color to an HSV color:
#-------------------------------------------------------------------------------

def rgb_to_hsv ( r, g, b, a ):
    # Get max and min RGB values:
    v = max( r, g, b )

    if v == 0.0:
        # r = g = b = 0    => s = 0, v is undefined
        return ( 0.0, 0.0, 0.0 )

    delta = v - min( r, g, b )

    if delta == 0.0:
        return ( 0.0, 0.0, v )

    if r == v:
        # h is between yellow and magenta:
        h = (g - b) / delta
    elif g == v:
        # h is between cyan and yellow:
        h = 2.0 + ((b - r) / delta)
    else:
        # h is between magenta and cyan:
        h = 4.0 + ((r - g) / delta)

    # Convert h to degrees and normalize it:
    h *= 60.0
    if h < 0.0:
        h += 360.0

    # Return the resulting HSV values:
    return ( h, delta / v, v )

