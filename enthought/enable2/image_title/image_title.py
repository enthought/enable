#-------------------------------------------------------------------------------
#
#  Define an image-based text title class.
#
#  Written by: David C. Morrill
#
#  Date: 10/10/2003
#
#  (c) Copyright 2003 by Enthought, Inc.
#
#  Classes defined: ImageTitle
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------
 
from enthought.traits.api               import RGBAColor, Trait, Str
from enthought.traits.ui.api            import Group, View, Include

from enthought.enable2.component     import Component
from enthought.enable2.enable_traits import alignment_trait, \
                                           font_trait, engraving_trait, \
                                           string_image_trait 
       
#-------------------------------------------------------------------------------
#  'ImageTitle' class:
#
#  Note: This class draws an image-based text title. The frame around the text
#        is comprised of 3 images, named "self.image + '_' + 0..2". The
#        images are drawn around the text as follows:
#
#        +---+--------------------+---+  
#        | 0 | 1 (text goes here) | 2 |  
#        +---+--------------------+---+
#
#        Image 1 is 'stretched' to fit the text as needed. If desired, image
#        numbering can be: 1..3.
#
#-------------------------------------------------------------------------------
        
class ImageTitle ( Component ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    image        = Trait( '=blue_round_3d_hs4', string_image_trait )
    text         = Str
    font         = font_trait
    color        = RGBAColor("black")
    shadow_color = RGBAColor("white")
    alignment    = alignment_trait
    style        = engraving_trait
     
    #---------------------------------------------------------------------------
    #  Trait view definitions:
    #---------------------------------------------------------------------------
    
    traits_view = View( Group( '<component>', id = 'component' ),
                        Group( '<links>',     id = 'links' ),
                        Group( 'image', 
                               id    = 'image', 
                               style = 'custom' ),
                        Group( 'text', ' ', 'font', '_', 'color', ' ',
                               'shadow_color', '_', 'alignment', ' ',
                               'style', 
                               id    = 'title', 
                               style = 'custom' ) )
    
    colorchip_map = {
       'fg_color':     'color',
       'shadow_color': 'shadow_color'
    }
    
    #---------------------------------------------------------------------------
    #  Initialize the object: 
    #---------------------------------------------------------------------------
    
    def __init__ ( self, text = '', **traits ):
        self.text = text
        Component.__init__( self, **traits )
        if self._images is None:
            self._image_changed( self.image )
        
    #---------------------------------------------------------------------------
    #  Handle the image name being changed: 
    #---------------------------------------------------------------------------
    
    def _image_changed ( self, image ):
        images = []
        for i in range( 4 ):
            try:
                images.append( self.image_for( '%s_%d' % ( image, i ) ) )
                if len( images ) == 3:
                    break
            except:
                if i > 0:
                    return
        self._images      = images
        self._image_sizes = ( images[0].height(), images[0].width(), 
                              images[1].width(),  images[2].width() )
       
        # Extract the y offset, if any:
        for i in range( -1, -len( image ) - 1, -1 ):
            if image[i] not in '-0123456789':
                break
        try:
            self._y_offset = int( image[ i + 1: ] )
        except:
            self._y_offset = 0
            
        self.layout()
        
    #---------------------------------------------------------------------------
    #  Handle various traits being changed: 
    #---------------------------------------------------------------------------
    
    def _text_changed ( self ):
        self.layout()
        
    def _font_changed ( self ):
        self.layout()
    
    def _color_changed ( self ):
        self.redraw()
    
    def _shadow_color_changed ( self ):
        self.redraw()
    
    def _alignment_changed ( self ):
        self.redraw()

    #---------------------------------------------------------------------------
    #  Lay out the contents of the component:
    #---------------------------------------------------------------------------
            
    def layout ( self ):
        if self._images is None:
            return
        if self.text == '':
            self._text_info = ( 0, 0, 0, 0 )
        else:
            gc = self.gc_temp()
            gc.set_font( self.font )
            self._text_info = gc.get_full_text_extent( self.text )
        self.min_width  = (self._text_info[0] + 
                           self._image_sizes[1] + self._image_sizes[3]) 
        self.min_height = self._image_sizes[0]
        
    #---------------------------------------------------------------------------
    #  Draw the image frame around the component: 
    #---------------------------------------------------------------------------
    
    def _draw ( self, gc ):
        gc.save_state()
        
        xl, yb, dx, dy     = self.bounds
        idy, ixl, ixm, ixr = self._image_sizes
        images             = self._images
        
        # Draw the two fixed-size image ends:
        gc.draw_image( images[0], ( xl, yb, ixl, idy ) ) 
        gc.draw_image( images[2], ( xl + dx  - ixr, yb, ixr, idy ) ) 
        
        # Draw all of the 'stretchable' images along the sides and middle:
        gc.stretch_draw( images[1], xl + ixl, yb, dx - ixl - ixr, idy ) 

        gc.text( self.text, ( xl + ixl, yb, dx - ixl - ixr, dy ), 
                 self.font, self.color_, self.shadow_color_, self.style_,
                 self.alignment_, self._y_offset, self._text_info )
        
        gc.restore_state()
            
    #---------------------------------------------------------------------------
    #  Generate any additional components that contain a specified (x,y) point:
    #---------------------------------------------------------------------------
       
    def _components_at ( self, x, y ):
        if self._image_at( x, y ) >= 0:
            return [ self ]
        return []
            
    #---------------------------------------------------------------------------
    #  Return the sub-image at a specified point:
    #---------------------------------------------------------------------------
       
    def _image_at ( self, x, y ):
        if not self.xy_in_bounds( x, y ):
            return -1

        xl, yb, dx, dy = self.bounds
        tx = x - xl
        ty = y - yb
        idy, ixl, ixm, ixr = self._image_sizes
        if ty >= idy:
            return -1
        xr = dx - ixr
        
        # Reduce x to valid image coordinates:
        if tx < ixl:
            index = 0
        elif tx < xr:
            index = 1
            tx   -= ixl
            tx   -= (int( tx ) / ixm) * ixm
        else:
            index = 2
            tx   -= xr
         
        # If the image alpha value of the point under (x,y) has a high alpha,
        # return the specified image index, otherwise indicate it is not over
        # an image:
        image = self._images[ index ]
        if ((image.bmp_array.shape[2] < 4) or
            (image.bmp_array[ idy - int( ty ) - 1, int( tx ), 3 ] >= 128)):
            return index
        return -1

