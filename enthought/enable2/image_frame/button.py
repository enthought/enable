#-------------------------------------------------------------------------------
#
#  Define an 'image' based Enable 'button' component.
#
#  Written by: David C. Morrill
#
#  Date: 10/09/2003
#
#  (c) Copyright 2003 by Enthought, Inc.
#
#  Classes defined: Button
#
#-------------------------------------------------------------------------------

import os.path

from enthought.traits.api               import Trait, Event, false, true
# ui_hack
#from enthought.traits.ui.api            import Group, View, Include
from enthought.enable.controls      import Label
from enthought.enable.radio_group   import RadioStyle, RadioGroup
from enthought.enable.enable_traits import string_image_trait
from image_frame                    import ImageFrame

#-------------------------------------------------------------------------------
#  'ButtonBase' class:
#-------------------------------------------------------------------------------

class ButtonBase ( ImageFrame ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    image    = Trait( '=trn3', string_image_trait )
    rollover = false
    enabled  = true

    # ui_hack
    ##---------------------------------------------------------------------------
    ##  Trait editor definition:
    ##---------------------------------------------------------------------------
    #
    #buttonbase_view = View(Include(id='imageframe_view'),
    #                       Group('rollover', 'enabled', label='Component'))

    #---------------------------------------------------------------------------
    #  Initialize the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, component = 'Button', **traits ):
        self._x_offset = self._y_offset = 0.0
        if isinstance(component, basestring):
            component = Label( component, padding_left  = 2, padding_right  = 2,
                                          padding_top   = 2, padding_bottom = 2,
                                          text_position = 'center' )
        ImageFrame.__init__( self, component, **traits )

    #---------------------------------------------------------------------------
    #  Select the image set to use:
    #---------------------------------------------------------------------------

    def _select_image ( self, type = 'n' ):
        image      = self.image
        base, ext  = os.path.splitext( image )
        self.image = '%s%s%s%s' % ( base[:-2], type, base[-1:], ext )

    #---------------------------------------------------------------------------
    #  Set the correct component offset:
    #---------------------------------------------------------------------------

    def _set_offset ( self, offset = 0.0 ):
        self._x_offset = -offset
        self._y_offset =  offset

    #---------------------------------------------------------------------------
    #  Generate any additional components that contain a specified (x,y) point:
    #---------------------------------------------------------------------------

    def _components_at ( self, x, y ):
        if ((self._image_at( x, y ) >= 0) or
            (len( self.component.components_at( x, y ) ) > 0)):
            return [ self ]
        return []

#-------------------------------------------------------------------------------
#  'Button' class:
#-------------------------------------------------------------------------------

class Button ( ButtonBase ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    clicked = Event( True )

    #---------------------------------------------------------------------------
    #  Mouse event handlers:
    #---------------------------------------------------------------------------

    def _left_down_changed ( self, event ):
        event.handled = self._down = True
        self.window.mouse_owner = self
        self._set_offset( self._offset )
        self._select_image( 'i' )

    def _left_dclick_changed ( self, event ):
        self._left_down_changed( event )

    def _left_up_changed ( self, event ):
        event.handled = True
        self._down    = None
        self._select_image()
        self._set_offset()
        if self._image_at( event ) >= 0:
            self.clicked = True
            if not self._rollover:
                self.window.mouse_owner = None
        else:
            self.window.mouse_owner = None
            if self._rollover:
                self._rollover = False
                self.redraw()

    def _mouse_move_changed ( self, event ):
        if self._down:
            event.handled = True
            index         = (self._image_at( event ) >= 0)
            self._select_image( 'ni'[ index ] )
            self._set_offset( [ 0.0, self._offset ][ index ] )
        elif self.rollover:
            if self._image_at( event ) >= 0:
                if not self._rollover:
                    self._rollover = True
                    self.window.mouse_owner = self
                    self.redraw()
            elif self._rollover:
                self._rollover = False
                self.window.mouse_owner = None
                self.redraw()

    #---------------------------------------------------------------------------
    #  Draw the component in a specified graphics context:
    #---------------------------------------------------------------------------

    def _draw ( self, gc ):
        if (not self.rollover) or self._rollover:
            self._pre_draw( gc )
        gc.save_state()
        gc.translate_ctm( self._x_offset, self._y_offset )
        self.component.draw( gc )
        gc.restore_state()
        self._post_draw( gc )

#-------------------------------------------------------------------------------
#  'CheckBoxButton' class:
#-------------------------------------------------------------------------------

class CheckBoxButton ( ButtonBase ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    selected = false

    # ui_hack
    ##---------------------------------------------------------------------------
    ##  Trait editor definition:
    ##---------------------------------------------------------------------------
    #
    #checkboxbutton_view = View(Include(id='buttonbase_view'),
    #                           Group( 'selected', label='Component'))

    #---------------------------------------------------------------------------
    #  Set up the correct image of shift offset to use:
    #---------------------------------------------------------------------------

    def _set_display ( self, index ):
        self._select_image( 'ni'[ index ] )
        self._set_offset( [ 0.0, self._offset ][ index ] )

    #---------------------------------------------------------------------------
    #  Handle the selected state being changed:
    #---------------------------------------------------------------------------

    def _selected_changed ( self ):
        self._set_display( self.selected )

    #---------------------------------------------------------------------------
    #  Set the selection state of the component:
    #---------------------------------------------------------------------------

    def _select ( self ):
        self.selected = not self.selected

    #---------------------------------------------------------------------------
    #  Mouse event handlers:
    #---------------------------------------------------------------------------

    def _left_down_changed ( self, event ):
        event.handled = self._down = True
        self.window.mouse_owner = self
        self._set_display( not self.selected )

    def _left_up_changed ( self, event ):
        event.handled = True
        self._down    = None
        if self._image_at( event ) >= 0:
            self._select()
            if self.selected or (not self._rollover):
                self.window.mouse_owner = None
                self._rollover = False
        else:
            self._selected_changed()
            self.window.mouse_owner = None
            if self._rollover:
                self._rollover = False
                self.redraw()

    def _mouse_move_changed ( self, event ):
        if self._down:
            event.handled = True
            self._set_display( (self._image_at( event ) >= 0) ^ self.selected )
        elif self.rollover:
            if self._image_at( event ) >= 0:
                if (not self.selected) and (not self._rollover):
                    self._rollover = True
                    self.window.mouse_owner = self
                    self.redraw()
            elif self._rollover:
                self._rollover = False
                self.window.mouse_owner = None
                self.redraw()

    #---------------------------------------------------------------------------
    #  Draw the component in a specified graphics context:
    #---------------------------------------------------------------------------

    def _draw ( self, gc ):
        if self.selected or (not self.rollover) or self._rollover:
            self._pre_draw( gc )
        gc.save_state()
        gc.translate_ctm( self._x_offset, self._y_offset )
        self.component.draw( gc )
        gc.restore_state()
        self._post_draw( gc )

#-------------------------------------------------------------------------------
#  'RadioButton' class:
#-------------------------------------------------------------------------------

class RadioButton ( CheckBoxButton, RadioStyle ):

    #---------------------------------------------------------------------------
    #  Set the selection state of the component:
    #---------------------------------------------------------------------------

    def _select ( self ):
        self.selected = True

    #---------------------------------------------------------------------------
    #  Handle the container the component belongs to being changed:
    #---------------------------------------------------------------------------

    def _container_changed ( self, old, new ):
        CheckBoxButton._container_changed( self )
        if self.radio_group is old.radio_group:
            self.radio_group = None
        if self.radio_group is None:
            if new.radio_group is None:
                new.radio_group = RadioGroup()
            new.radio_group.add( self )

    #---------------------------------------------------------------------------
    #  Handle the 'selected' status of the checkbox being changed:
    #---------------------------------------------------------------------------

    def _selected_changed ( self ):
        CheckBoxButton._selected_changed( self )
        if self.selected:
            self.radio_group.selection = self

