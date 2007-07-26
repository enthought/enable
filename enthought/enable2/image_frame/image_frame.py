#-------------------------------------------------------------------------------
#
#  Define 'image' based Enable 'frame' components.
#
#  Written by: David C. Morrill
#
#  Date: 10/09/2003
#
#  (c) Copyright 2003 by Enthought, Inc.
#
#  Classes defined: ImageFrame
#                   TitleFrame
#                   ResizeFrame
#                   WindowFrame
#                   ComponentFactory
#
#  ToDo: - ImageFrame: Set the frame 'min_size' values based on the image
#          sizes and the inner components 'min_size' values.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.api       import Trait, Str, true
from enthought.traits.ui.api    import Group, View, Include

from enthought.enable.base  import add_rectangles
from enthought.enable.frame import Frame
from enthought.enable.enable_traits import alignment_trait, font_trait, engraving_trait, \
     string_image_trait
from enthought.enable2.traits.rgba_color_trait import RGBAColor


#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

title_inset = ( 8, 0, -16, 0 )

#-------------------------------------------------------------------------------
#  'ImageFrame' class:
#
#  Note: This class draws an image-based frame around its inner component. The
#        frame is comprised of 9 images, named "self.image + '_' + 0..8". The
#        images are drawn around the frame as follows:
#
#        +-----+---+------+
#        | 0   | 1 |  2   |
#        +---+-+---+--+---+
#        | 3 |   4    | 5 |
#        +---++----+------+
#        | 6  | 7  |  8   |
#        +----+----+------+
#
#        Images 1, 3, 4, 5, 7 are 'stretched' to fit around the sides of the
#        inner component as needed. Image 4 can be ommitted if desired. Also,
#        image numbering can begin at 1 instead of 0.
#
#-------------------------------------------------------------------------------

class ImageFrame ( Frame ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    image = Trait( '=tsi2', string_image_trait )

    #---------------------------------------------------------------------------
    #  Trait view definitions:
    #---------------------------------------------------------------------------

    traits_view = View( Group( '<component>', id = 'component' ),
                        Group( '<links>',     id = 'links' ),
                        Group( 'image',
                               id    = 'image',
                               style = 'custom' ) )

    #---------------------------------------------------------------------------
    #  Initialize the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, component = None, **traits ):
        Frame.__init__( self, component, **traits )
        self._image_changed( self.image )

    #---------------------------------------------------------------------------
    #  Handle the image file root name being changed:
    #---------------------------------------------------------------------------

    def _image_changed ( self, image ):
        images     = []
        first      = 0
        if image.find( '%d' ) >= 0:
            image_name = image
            image      = image.replace( '%d', '' )
        else:
            image_name = image + '_%d'
        for i in range( 10 ):
            try:
                images.append( self.image_for( image_name % i ) )
                if len( images ) == 9:
                    break
            except:
                if i == (first + 4):
                    images.append( None )
                elif i == 0:
                    first = 1
                else:
                    return
        self._images      = images
        self._image_sizes = [ images[0].width(),  images[1].width(),
                              images[2].width(),  images[3].width(),
                              1,                  images[5].width(),
                              images[6].width(),  images[7].width(),
                              images[8].width(),  images[0].height(),
                              images[3].height(), images[6].height() ]
        if images[4] is not None:
            self._image_sizes[4] = images[4].width()

        # Extract the y offset, if any:
        col = image.rfind( '.' )
        if col >= 0:
            image = image[:col]
        while image[-1:] == '_':
            image = image[:-1]
        for i in range( -1, -len( image ) - 1, -1 ):
            if image[i] not in '-0123456789':
                break
        try:
            self._offset = int( image[ i + 1: ] )
        except:
            self._offset = 0

        self._bounds_modified( self.bounds )
        self._component_min_size_modified()
        self.redraw()

    #---------------------------------------------------------------------------
    #  Draw the image frame around the component:
    #---------------------------------------------------------------------------

    def _pre_draw ( self, gc ):
        gc.save_state()

        xl, yb, dx, dy = self.bounds
        gc.clip_to_rect(xl, yb, dx, dy)
        ( ixlt, ixmt, ixrt, ixlm, ixmm, ixrm, ixlb, ixmb, ixrb,
          iyt, iym, iyb ) = self._image_sizes
        yt     = yb + dy  - iyt
        ym     = yb + iyb
        dym    = dy - iyt - iyb
        images = self._images

        # Draw the four fixed-size image corners:
        gc.draw_image( images[0], ( xl, yt, ixlt, iyt ) )
        gc.draw_image( images[2], ( xl + dx - ixrt, yt, ixrt, iyt ) )
        gc.draw_image( images[6], ( xl, yb, ixlb, iyb ) )
        gc.draw_image( images[8], ( xl + dx - ixrb, yb, ixrb, iyb ) )

        # Draw all of the 'stretchable' images along the sides and middle:
        gc.stretch_draw( images[1], xl + ixlt, yt, dx - ixlt - ixrt, iyt )
        gc.stretch_draw( images[3], xl, ym, ixlm, dym )
        gc.stretch_draw( images[5], xl + dx - ixrm, ym, ixrm, dym )
        gc.stretch_draw( images[7], xl + ixlb, yb, dx - ixlb - ixrb, iyb )

        # Only draw the center region if we have an image for it:
        if images[4] is not None:
            gc.stretch_draw( images[4], xl + ixlm, ym, dx - ixlm - ixrm, dym )

        gc.restore_state()

    #---------------------------------------------------------------------------
    #  Generate any additional components that contain a specified (x,y) point:
    #---------------------------------------------------------------------------

    def _components_at ( self, x, y ):
        if self._image_at( x, y ) >= 0:
            return [ self ] + self.component.components_at( x, y )
        return self.component.components_at( x, y )

    #---------------------------------------------------------------------------
    #  Return the sub-image at a specified point:
    #---------------------------------------------------------------------------

    def _image_at ( self, x, y = None ):
        # Allow ( x = mouse_event, y = None ) case:
        if y is None:
            y = x.y
            x = x.x

        if not self.xy_in_bounds( x, y ):
            return -1

        xl, yb, dx, dy = self.bounds
        tx = x - xl
        ty = y - yb
        ( ixlt, ixmt, ixrt, ixlm, ixmm, ixrm, ixlb, ixmb, ixrb,
          iyt, iym, iyb ) = self._image_sizes
        yt = dy - iyt

        # Reduce (x,y) to valid image coordinates:
        if ty < iyb:
            idy = iyb
            if tx < ixlb:
                index = 6
                tx    = int( tx )
            elif tx < (dx - ixrb):
                index = 7
                tx    = int( tx - ixlb ) % ixmb
            else:
                index = 8
                tx    = int( tx - (dx - ixrb) )
        elif ty < yt:
            idy = iym
            ty  = int( ty - iyb ) % iym
            if tx < ixlm:
                index = 3
                tx    = int( tx )
            elif tx < (dx - ixrm):
                index = 4
                tx    = int( tx - ixlm ) % ixmm
            else:
                index = 5
                tx    = int( tx - (dx - ixrm) )
        else:
            idy = iyt
            ty -= yt
            if tx < ixlt:
                index = 0
                tx    = int( tx )
            elif tx < (dx - ixrt):
                index = 1
                tx    = int( tx - ixlt ) % ixmt
            else:
                index = 2
                tx    = int( tx - (dx - ixrt) )

        # If the image alpha value of the point under (x,y) has a high alpha,
        # return the specified image index, otherwise indicate it is not over
        # an image:
        image = self._images[ index ]
        if ((image is not None) and ((image.bmp_array.shape[2] < 4) or
            (image.bmp_array[ idy - int( ty ) - 1, int( tx ), 3 ] >= 128))):
            return index
        return -1

    #---------------------------------------------------------------------------
    #  Determine if component is properly initialized:
    #---------------------------------------------------------------------------

    def initialized ( self ):
        return ((self.component is not None) and
                (self._image_sizes is not None))

    #---------------------------------------------------------------------------
    #  Handle the bounds of the outer component being changed:
    #---------------------------------------------------------------------------

    def _bounds_modified ( self, bounds ):
        if self.initialized():
            x, y, dx, dy = bounds
            ( ixlt, ixmt, ixrt, ixlm, ixmm, ixrm, ixlb, ixmb, ixrb,
              iyt, iym, iyb ) = self._image_sizes
            self.component.bounds = ( x + ixlm, y + iyb,
                                      dx - ixlm - ixrm, dy - iyb - iyt )

    #---------------------------------------------------------------------------
    #  Handle the bounds of the inner component being modified:
    #---------------------------------------------------------------------------

    def _component_bounds_modified ( self, bounds ):
        if self._image_sizes is not None:
            x, y, dx, dy = bounds
            ( ixlt, ixmt, ixrt, ixlm, ixmm, ixrm, ixlb, ixmb, ixrb,
              iyt, iym, iyb ) = self._image_sizes
            self.bounds = ( x - ixlm, y - iyb,
                            dx + ixlm + ixrm, dy + iyb + iyt )

    #---------------------------------------------------------------------------
    #  Handle the minimum size of the component we are bound to being modified:
    #---------------------------------------------------------------------------

    def _component_min_size_modified ( self ):
        if self.initialized():
            ( ixlt, ixmt, ixrt, ixlm, ixmm, ixrm, ixlb, ixmb, ixrb,
              iyt, iym, iyb ) = self._image_sizes
            self.min_width  = max( self.component.min_width  + ixlm + ixrm,
                                   ixlt + ixrt, ixlb + ixrb )
            self.min_height = self.component.min_height + iyt + iyb

#-------------------------------------------------------------------------------
#  'ResizeFrame' class
#-------------------------------------------------------------------------------

class ResizeFrame ( Frame ):

    #---------------------------------------------------------------------------
    #  Do any drawing that needs to be done after drawing the contained
    #  component:
    #---------------------------------------------------------------------------

    def _post_draw ( self, gc ):
        resize = self._resize
        if resize is None:
            self._resize = resize = self.image_for( '=resize' )
        x, y, dx, dy = self.bounds
        idx = resize.width()
        gc.draw_image( resize, ( x + dx - idx, y, idx, resize.height() ) )

    #---------------------------------------------------------------------------
    #  Check whether the pointer is over the resize area:
    #---------------------------------------------------------------------------

    def _over_resize ( self, x, y ):
        xl, yb, dx, dy = self.bounds
        tx = x - (xl + dx - self._resize.width())
        ty = y - yb
        return ((tx >= 0.0) and (ty >= 0.0) and (tx >= ty))

    #---------------------------------------------------------------------------
    #  Generate the components that contain a specified (x,y) point:
    #---------------------------------------------------------------------------

    def _components_at ( self, x, y ):
        if self._over_resize( x, y ):
            return [ self ]
        return self.component.components_at( x, y )

    #---------------------------------------------------------------------------
    #  Handle a resizing operation:
    #---------------------------------------------------------------------------

    def _left_down_changed ( self, event ):
        self._x, self._y        = event.x, event.y
        self.window.mouse_owner = self
        event.handled           = True

    def _left_up_changed ( self, event ):
        self._x = self._y = None
        self.pointer = 'arrow'
        self.window.mouse_owner = None

    def _mouse_move_changed ( self, event ):
        if self._x is not None:
            comp   = self.component
            resize = self._resize
            dx     = event.x - self._x
            dy     = event.y - self._y
            if (dx != 0) or (dy != 0):
                cx, cy, cdx, cdy = self.bounds
                ndx = max( cdx + dx, comp.min_width,  resize.width()  )
                ndy = max( cdy - dy, comp.min_height, resize.height() )
                dx  = ndx - cdx
                dy  = cdy - ndy
                self.bounds = ( cx, cy + dy, ndx, ndy )
                self._x += dx
                self._y += dy
        elif self._over_resize( event.x, event.y ):
            self.pointer = 'size bottom right'
            self.window.mouse_owner = self
        else:
            self.pointer = 'arrow'
            self.window.mouse_owner = None

#-------------------------------------------------------------------------------
#  'TitleFrame' class
#-------------------------------------------------------------------------------

class TitleFrame ( ImageFrame ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    image        = Trait( '=window', string_image_trait )
    title        = Str
    font         = Trait( 'modern 9', font_trait )
    color        = RGBAColor("white")
    shadow_color = RGBAColor("black")
    alignment    = Trait( 'left', alignment_trait )
    style        = Trait( 'engraved', engraving_trait )

    #---------------------------------------------------------------------------
    #  Trait view definitions:
    #---------------------------------------------------------------------------

    traits_view = View( Group( '<component>', id = 'component' ),
                        Group( '<links>',     id = 'links' ),
                        Group( '<image>',     id = 'image' ),
                        Group( 'title', ' ', 'font', '_', 'color', ' ',
                               'shadow_color', '_', 'alignment', 'style',
                               id    = 'title',
                               style = 'custom' ) )

    colorchip_map = {
       'fg_color':     'color',
       'shadow_color': 'shadow_color'
    }

    #---------------------------------------------------------------------------
    #  Initialize the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, component = None, **traits ):
        self._title_info = ( 0, 0, 0, 0 )
###        self.path        = self
        ImageFrame.__init__( self, component, **traits )

    #---------------------------------------------------------------------------
    #  Handle the title text being changed:
    #---------------------------------------------------------------------------

    def _title_changed ( self, title ):
        if title == '':
            self._title_info = ( 0, 0, 0, 0 )
        else:
            gc = self.gc_temp()
            gc.set_font( self.font )
            self._title_info = gc.get_full_text_extent( title )

    #---------------------------------------------------------------------------
    #  Draw the image frame around the component:
    #---------------------------------------------------------------------------

    def _pre_draw ( self, gc ):
        ImageFrame._pre_draw( self, gc )

        if self.title == '':
            return

        gc.save_state()
        gc.clip_to_rect(*add_rectangles(self.bounds, title_inset))
        #( ixlt, ixmt, ixrt, ixlm, ixmm, ixrm, ixlb, ixmb, ixrb,
        #   iyt, iym, iyb ) = self._image_sizes
        ixlt = self._image_sizes[0]
        iyt = self._image_sizes[9]
        ixrt = self._image_sizes[2]
        xl, yb, dx, dy    = self.bounds
        gc.text( self.title,
                 ( xl + ixlt, yb + dy - iyt, dx - ixlt - ixrt, iyt ),
                 self.font, self.color_, self.shadow_color_, self.style_,
                 self.alignment_, -self._offset, self._title_info )
        gc.restore_state()

#-------------------------------------------------------------------------------
#  'WindowFrame' class
#-------------------------------------------------------------------------------

class WindowFrame ( TitleFrame ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    movable   = true
    resizable = true

    #---------------------------------------------------------------------------
    #  Trait view definitions:
    #---------------------------------------------------------------------------

    traits_view = View( Group( '<component>', 'movable', 'resizable',
                               id = 'component' ),
                        Group( '<links>', id = 'links' ),
                        Group( '<image>', id = 'image' ),
                        Group( '<title>', id = 'title' ) )

    pointer_map = {
       0: ( 'size top left',     ( 1, 0, -1,  1 ) ),
       1: ( 'size top',          ( 0, 0,  0,  1 ) ),
       2: ( 'size top right',    ( 0, 0,  1,  1 ) ),
       3: ( 'size left',         ( 1, 0, -1,  0 ) ),
       5: ( 'size right',        ( 0, 0,  1,  0 ) ),
       6: ( 'size bottom left',  ( 1, 1, -1, -1 ) ),
       7: ( 'size bottom',       ( 0, 1,  0, -1 ) ),
       8: ( 'size bottom right', ( 0, 1,  1, -1 ) )
    }

    #---------------------------------------------------------------------------
    #  Handle a resize or move operation:
    #---------------------------------------------------------------------------

    def _left_down_changed ( self, event ):
        event.handled = True
        if not (self.resizable or self.movable):
            return

        pointer, factors = self._pointer_for( event )
        if pointer is not None:
            if pointer == 'arrow':
                if self.movable:
                    self.window.mouse_owner = None
                    self.window.drag( self, self.container, event,
                                      alpha = -1.0 )
            elif self.resizable:
                self._factors           = factors
                self._x, self._y        = event.x, event.y
                self.window.mouse_owner = self

    def _left_up_changed ( self, event ):
        self._factors = self._x = self._y = None
        self.pointer  = 'arrow'
        self.window.mouse_owner = None

    def _mouse_move_changed ( self, event ):
        if self._factors is not None:
            dx, dy           = (event.x - self._x), (event.y - self._y)
            cx, cy, cdx, cdy = self.bounds
            fx, fy, fdx, fdy = self._factors
            ncdx = cdx + fdx * dx
            if ncdx < self.min_width:
                ncdx = self.min_width
                dx   = (self.min_width - cdx) / fdx
            ncdy = cdy + fdy * dy
            if ncdy < self.min_height:
                ncdy = self.min_height
                dy   = (self.min_height - cdy) / fdy
            self.bounds = (  cx + fx * dx,  cy + fy * dy, ncdx, ncdy )
            self._x += dx
            self._y += dy
        else:
            if self.resizable:
                pointer, factors = self._pointer_for( event )
            else:
                pointer = None
            if pointer is None:
                self.pointer            = 'arrow'
                self.window.mouse_owner = None
            else:
                self.pointer            = pointer
                self.window.mouse_owner = self
                event.handled           = True

    #---------------------------------------------------------------------------
    #  Determine the pointer and handler name for a specified point:
    #---------------------------------------------------------------------------

    def _pointer_for ( self, event ):
        x, y     = event.x, event.y
        quadrant = self._image_at( x, y )
        pointer, factors = self.pointer_map.get( quadrant, ( None, None ) )
        if (quadrant == 1) and ((self.top - 4) > y):
            pointer = 'arrow'
        return ( pointer, factors )

#-------------------------------------------------------------------------------
#  'ComponentFactory' class:
#-------------------------------------------------------------------------------

class ComponentFactory ( ImageFrame ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    image = Trait( '=tsi3', string_image_trait )

    #---------------------------------------------------------------------------
    #  Initialize the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, factory,
                         args      = None,
                         traits    = None,
                         component = None,
                         **trait_dict ):
        self._factory = factory
        self._args    = args   or ()
        self._traits  = traits or {}
        if component is None:
            component = self.create_component()
        ImageFrame.__init__( self, component, **trait_dict )

    #---------------------------------------------------------------------------
    #  Create an instance of the component that the factory produces:
    #---------------------------------------------------------------------------

    def create_component ( self ):
        factory = self._factory
        if isinstance(factory, basestring):
            return eval( factory )
        return factory( *self._args, **self._traits )

    #---------------------------------------------------------------------------
    #  Generate any additional components that contain a specified (x,y) point:
    #---------------------------------------------------------------------------

    def _components_at ( self, x, y ):
        if self._image_at( x, y ) >= 0:
            return [ self ]
        return []

    #---------------------------------------------------------------------------
    #  Handle mouse events:
    #---------------------------------------------------------------------------

    def _left_down_changed ( self, event ):
        event.handled = True
        self.window.drag( self, self.container, event,
                          drag_copy = True, alpha = -1.0 )

