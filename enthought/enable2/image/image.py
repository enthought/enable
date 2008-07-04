#-------------------------------------------------------------------------------
#
#  Define standard Enable 'image' based control components.
#
#  Written by: David C. Morrill
#
#  Date: 10/10/2003
#
#  (c) Copyright 2003 by Enthought, Inc.
#
#  Classes defined: Image
#                   Inspector
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.api import Trait, TraitPrefixList
from enthought.traits.ui.api import Group, View

from enthought.enable2.base import IDroppedOnHandler, gc_image_for
from enthought.enable2.component import Component
from enthought.enable2.enable_traits import string_image_trait, NoStretch
from enthought.enable2.colors import ColorTrait

#-------------------------------------------------------------------------------
#  'Image' class:
#-------------------------------------------------------------------------------

class Image ( Component ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    image          = Trait( '=image', string_image_trait )
    stretch_width  = NoStretch
    stretch_height = NoStretch

    #---------------------------------------------------------------------------
    #  Trait view definitions:
    #---------------------------------------------------------------------------

    traits_view = View( Group( '<component>', 'image', id = 'component' ),
                        Group( '<links>',              id = 'links' ) )

    #---------------------------------------------------------------------------
    #  Initialize the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, **traits ):
        Component.__init__( self, **traits )
        self._image_changed( self.image )

    #---------------------------------------------------------------------------
    #  Return an image specified by name:
    #---------------------------------------------------------------------------

    def image_for ( self, image ):
        path   = ''
        prefix = image[:1]
        if prefix == '=':
            path  = self
            image = image[1:]
        elif prefix == '.':
            path  = None
            image = image[1:]
        return gc_image_for( image, path )

    #---------------------------------------------------------------------------
    #  Draw the component in a specified graphics context:
    #---------------------------------------------------------------------------

    def _draw ( self, gc ):
        gc.draw_image( self._image, self.bounds )

    #---------------------------------------------------------------------------
    #  Handle the image being changed:
    #---------------------------------------------------------------------------

    def _image_changed ( self, image ):
        self._image     = image = self.image_for( image )
        self.width = image.width()
        self.height = image.height()

    #---------------------------------------------------------------------------
    #  Return the components that contain a specified (x,y) point:
    #---------------------------------------------------------------------------

    def _components_at ( self, x, y ):
        bmp = self._image.bmp_array
        if ((bmp.shape[2] < 4) or
            (bmp[ int( self.y + self.height - y ) - 1,
                  int( x - self.x ), 3 ] >= 128)):
            return [ self ]
        return []

#-------------------------------------------------------------------------------
#  'DraggableImage' class:
#-------------------------------------------------------------------------------

class DraggableImage ( Image ):

    #---------------------------------------------------------------------------
    #  Allow a copy of the image to be dragged:
    #---------------------------------------------------------------------------

    def _left_down_changed ( self, event ):
        event.handled = True
        self.window.drag( self, None, event, True, alpha = -1.0 )

#-------------------------------------------------------------------------------
#  'Inspector' class:
#-------------------------------------------------------------------------------

class Inspector ( DraggableImage, IDroppedOnHandler ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    image = Trait( '=inspector', string_image_trait )

    #---------------------------------------------------------------------------
    #  Handle being dropped on a component:
    #---------------------------------------------------------------------------

    def was_dropped_on ( self, component, event ):
        event.handled = True
        component.edit_traits( kind = 'live' )

#-------------------------------------------------------------------------------
#  'ColorChip' class:
#-------------------------------------------------------------------------------

class ColorChip ( Component ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    color = ColorTrait("yellow")
    item  = Trait( 'fg_color', TraitPrefixList(
                 [ 'fg_color',     'bg_color',
                   'shadow_color', 'alt_color' ] ) )

    #---------------------------------------------------------------------------
    #  Trait view definitions:
    #---------------------------------------------------------------------------

    traits_view = View( Group( '<component>', id = 'component' ),
                        Group( '<links>',     id = 'links' ),
                        Group( 'color',
                               id     = 'color',
                                style = 'custom' ) )

    #---------------------------------------------------------------------------
    #  Initialize the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, **traits ):
        Component.__init__( self, **traits )
        self._item      = self.item
        self._image     = image = self.image_for( '=colorchip' )
        self.min_width  = image.width()
        self.min_height = image.height()
        self.dimensions( self.min_width, self.min_height )

    #---------------------------------------------------------------------------
    #  Draw the component in a specified graphics context:
    #---------------------------------------------------------------------------

    def _draw ( self, gc ):
        gc.save_state()
        x, y, dx, dy = self.bounds
        gc.set_fill_color( self.color_ )
        gc.begin_path()
        gc.rect( x + 1, y + 1, dx - 2, dy - 2 )
        gc.fill_path()
        gc.draw_image( self._image, self.bounds )
        gc.restore_state()

    #---------------------------------------------------------------------------
    #  Mouse event handlers:
    #---------------------------------------------------------------------------

    def _left_down_changed ( self, event ):
        self.item = self._item
        if event.shift_down:
            if event.control_down:
                self.item = 'shadow_color'
            else:
                self.item = 'fg_color'
        elif event.control_down:
            self.item = 'bg_color'
        elif event.alt_down:
            self.item = 'alt_color'
        self.window.drag( self, None, event, True )
        event.handled = True

    def _right_up_changed ( self, event ):
        event.handled = True
        self.edit_traits( view = View( 'color@' ), kind = 'live' )

