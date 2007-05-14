#-------------------------------------------------------------------------------
#
#  Create a Tkinter-based window that contains some simple Enable components and
#  containers as well as some Chaco objects.
#
#  Written by: David C. Morrill
#
#  Date: 10/02/2003
#
#  (c) Copyright 2003 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.util.numerix         import arange
from enthought.enable.enable_traits import white_color_trait, black_color_trait
from enthought.tkenable import Component, ImageTitle, Inspector, Image, \
                               Container, ImageFrame, ResizeFrame, WindowFrame,\
                               Window
from enthought.enable.base          import bounds_to_coordinates, \
     coordinates_to_bounds, add_rectangles, half_pixel_bounds_inset, \
     gc_image_for, xy_in_bounds
from enthought.chaco.plot_component import PlotComponent
from enthought.chaco.plot_canvas    import PlotGroup, PlotCanvas, PlotIndexed
from enthought.chaco.plot_frame     import PlotTitle, PlotOverlay, PlotAxis
from enthought.chaco.plot_value     import PlotValue
from enthought.traits.api               import TraitGroup
                          
#-------------------------------------------------------------------------------
#  'Box' class:
#-------------------------------------------------------------------------------

class Box ( Component ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    border_color = black_color_trait
    fill_color   = white_color_trait
     
    #---------------------------------------------------------------------------
    #  Trait editor definition:
    #---------------------------------------------------------------------------
        
    __editable_traits__ = ( Component.__editable_traits__ + 
                            TraitGroup( 'border_color', '-', 'fill_color',
                                        label = 'Box',
                                        style = 'custom' ) )
    
    #---------------------------------------------------------------------------
    #  Draw the component in a specified graphics context:
    #---------------------------------------------------------------------------
    
    def _draw ( self, gc ):
        gc.save_state()
        gc.set_line_width( 1 )
        gc.set_stroke_color( self.border_color_ )
        if self._selected is not None:
            gc.set_fill_color( self._selected )
        else:
            gc.set_fill_color( self.fill_color_ )
        gc.begin_path()
        gc.rect( *add_rectangles( self.bounds, half_pixel_bounds_inset ) )
        gc.draw_path()
        gc.restore_state()
        
    #---------------------------------------------------------------------------
    #  Handle the user clicking the left mouse button:
    #---------------------------------------------------------------------------
        
    def _left_down_changed ( self, event ):
        self.window.drag( self, [ self.container, None ][ event.control_down ], 
                          event ) 
        event.handled = True
        
    def _right_down_changed ( self, event ):
        self.window.drag( self, [ self.container, None ][ event.control_down ], 
                          event, True ) 
        event.handled = True
        
    #---------------------------------------------------------------------------
    #  Handle dropping the object:
    #---------------------------------------------------------------------------
        
    def drag_enter_by_mycolorchip ( self, colorchip, event ):
        if not colorchip._set_border:
            self._selected = colorchip.fill_color
            self.redraw()
            
    def drag_leave_by_mycolorchip ( self, colorchip, event ):
        if not colorchip._set_border:
            self._selected = None
            self.redraw()
        
    def dropped_on_by_mycolorchip ( self, colorchip, event ):
        event.handled  = True
        self._selected = None
        if colorchip._set_border:
            self.border_color = colorchip.fill_color
        else:
            self.fill_color = colorchip.fill_color

#-------------------------------------------------------------------------------
#  'MyColorChip' class:
#-------------------------------------------------------------------------------

class MyColorChip ( Box ):
        
    #---------------------------------------------------------------------------
    #  Handle the user clicking the left mouse button:
    #---------------------------------------------------------------------------
        
    def _left_down_changed ( self, event ):
        self._set_border = False
        self.window.drag( self, None, event, True ) 
        event.handled = True
        
    def _right_down_changed ( self, event ):
        self._set_border = True
        self.window.drag( self, None, event, True ) 
        event.handled = True
        
    #---------------------------------------------------------------------------
    #  Handle dropping the object:
    #---------------------------------------------------------------------------
        
    def _drag_enter_changed ( self, event ):
        pass
            
    def _drag_leave_changed ( self, event ):
        pass
        
    def _dropped_on_changed ( self, event ):
        pass
                          
#-------------------------------------------------------------------------------
#  'Image' class:
#-------------------------------------------------------------------------------

class MyImage ( Image ):
        
    #---------------------------------------------------------------------------
    #  Handle the user clicking the left mouse button:
    #---------------------------------------------------------------------------
        
    def _left_down_changed ( self, event ):
        self.window.drag( self, [ self.container, None ][ event.control_down ], 
                          event, alpha = -1.0 ) 
        event.handled = True
        
    def _right_down_changed ( self, event ):
        self.window.drag( self, [ self.container, None ][ event.control_down ], 
                          event, True, alpha = -1.0 ) 
        event.handled = True
        
    def _mouse_move_changed ( self, event ):
        event.handled = True

#-------------------------------------------------------------------------------
#  'Profiler' class:
#-------------------------------------------------------------------------------
        
class Profiler ( Image ):
    
    def _left_down_changed ( self, event ):
        event.handled = True
    
    def _mouse_move_changed ( self, event ):
        event.handled = True
        
    def _left_up_changed ( self, event ):
        import enthought.gotcha as gotcha
        if self.image == 'profiler_off':
            self.image = 'profiler_on'
            gotcha.begin_profiling()
        else:
            self.image = 'profiler_off'
            gotcha.end_profiling()
        event.handled = True
        
#-------------------------------------------------------------------------------
#  'SimpleContainer' class:  
#-------------------------------------------------------------------------------        

class SimpleContainer ( Container ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    border_color = black_color_trait
    fill_color   = white_color_trait
     
    #---------------------------------------------------------------------------
    #  Trait editor definition:
    #---------------------------------------------------------------------------
        
    __editable_traits__ = ( Container.__editable_traits__ +
                            TraitGroup( 'border_color', '-', 'fill_color',
                                        label = 'Container',
                                        style = 'custom' ) )
                
    #---------------------------------------------------------------------------
    #  Draw the container background in a specified graphics context:
    #---------------------------------------------------------------------------
    
    def _draw_container ( self, gc ):
        gc.save_state()
        gc.set_line_width( 1 )
        gc.set_stroke_color( self.border_color_ )
        gc.set_fill_color( self.fill_color_ )
        gc.begin_path()
        gc.rect( *add_rectangles( self.bounds, half_pixel_bounds_inset ) )
        gc.draw_path()
        gc.restore_state()
            
    #---------------------------------------------------------------------------
    #  Allow the container and its components to be dragged: 
    #---------------------------------------------------------------------------
        
    def _left_down_changed ( self, event ):
        self.window.drag( self, None, event ) 
        event.handled = True
        
    def _dropped_changed ( self, event ):
        self.location( event.x, event.y )
    
    #---------------------------------------------------------------------------
    #  Handle being dropped on by various object types: 
    #---------------------------------------------------------------------------
                
    def dropped_on_by_mycolorchip ( self, colorchip, event ):
        event.handled = True
        if colorchip._set_border:
            self.border_color = colorchip.fill_color
        else:
            self.fill_color = colorchip.fill_color
            
    def dropped_on_by_box ( self, box, event ):
        event.handled = True
        x, y, dx, dy  = box.bounds
        if event.copy:
            box = Box( fill_color   = box.fill_color,
                       border_color = box.border_color )
        box.bounds = ( x + event.x - event.x0, y + event.y - event.y0, dx, dy )
        self._check_bounds( box )
        self.add( box )
            
    def dropped_on_by_myimage ( self, image, event ):
        event.handled = True
        x, y, dx, dy  = image.bounds
        if event.copy:
            image = MyImage( image = image.image )
        image.bounds = ( x + event.x - event.x0, y + event.y - event.y0, dx, dy )
        self._check_bounds( image )
        self.add( image )
                
    def dropped_on_by_plotcomponent ( self, pc, event ):
        event.handled = True
        x, y, dx, dy  = pc.bounds
        if event.copy:
            original = pc.component
            clone    = original.__class__()
            clone.clone_traits( original )
            pc = PlotComponent( clone )
            clone.on_trait_change( dropped_on_changed, 'dropped_on' )
            pc.on_trait_change( left_down_changed,  'left_down' )
            pc.on_trait_change( right_down_changed, 'right_down')
            if isinstance( original, PlotValue ):
                clone.on_trait_change( drag_enter_changed, 'drag_enter' )
                clone.on_trait_change( drag_leave_changed, 'drag_leave' )
        pc.bounds = ( x + event.x - event.x0, y + event.y - event.y0, dx, dy )
        self._check_bounds( pc )
        self.add( pc )
                
    def dropped_on_by_lithologycomponent ( self, lc, event ):
        event.handled = True
        x, y, dx, dy  = lc.bounds
        if event.copy:
            lc = LithologyComponent( model = lc.model )
            lc.on_trait_change( left_down_changed,  'left_down' )
            lc.on_trait_change( right_down_changed, 'right_down' )
            lc.on_trait_change( middle_up_changed,  'middle_up' )
        lc.bounds = ( x + event.x - event.x0, y + event.y - event.y0, dx, dy )
        self._check_bounds( lc )
        self.add( lc )
            
    def dropped_on_by_frame ( self, frame, event ):
        event.handled = True
        x, y, dx, dy  = frame.bounds
        frame.bounds = ( x + event.x - event.x0, y + event.y - event.y0, dx, dy)
        self._check_bounds( frame )
        self.add( frame )

#-------------------------------------------------------------------------------
#  'EnableWindowFrame' class:
#-------------------------------------------------------------------------------
       
#-------------------------------------------------------------------------------
#  Handle the user clicking the left mouse button:
#-------------------------------------------------------------------------------

def draggable ( component ):
    component.on_trait_change( do_drag,         'left_down' )
    component.on_trait_change( do_clone,        'right_down' )
    component.on_trait_change( dropped_changed, 'dropped' )
    return component
    
def do_drag ( object, name, event ):
    object.window.drag( object, 
                        [ object.container, None ][event.control_down], 
                        event, alpha = -1.0 ) 
    event.handled = True
    
def do_clone ( object, name, event ):
    object.window.drag( object, 
                        [ object.container, None ][event.control_down], 
                        event, True, alpha = -1.0 ) 
    event.handled = True
    
def left_down_changed ( object, name, event ):
    if not event.handled:
        object.window.drag( object, 
                            [ object.container, None ][event.control_down], 
                            event ) 
        event.handled = True
    
def left_down_changed_no_alpha ( object, name, event ):
    if not event.handled:
        object.window.drag( object, 
                            [ object.container, None ][ event.control_down ], 
                            event, alpha = -1.0 ) 
        event.handled = True
    
def right_down_changed ( object, name, event ):
    if not event.handled:
        object.window.drag( object, 
                            [ object.container, None ][event.control_down], 
                            event, True ) 
        event.handled = True
        
def middle_up_changed ( object, name, event ):
    object.model.selected = 1 - object.model.selected
    object.redraw()
    
def dropped_changed ( object, name, event ):
    event.handled = True
    if event.copy:
        clone = object.__class__()
        clone.clone_traits( object )
        object.container.add( clone )
        object = draggable( clone )
    object.location( event.x, event.y )    

#-------------------------------------------------------------------------------
#  Handle 'drag_enter' and 'drag_leave' events:  
#-------------------------------------------------------------------------------

def drag_enter_changed ( object, name, event ):
    component = event.components[0]
    if isinstance( component, MyColorChip ) and (not component._set_border):
        object._old_line_color = object.line_color
        object.line_color      = component.fill_color
        object.redraw()
        
def drag_leave_changed ( object, name, event ):
    component = event.components[0]
    if isinstance( component, MyColorChip ) and (not component._set_border):
        object.line_color = object._old_line_color
        object.redraw()
     
#-------------------------------------------------------------------------------
#  Handle dropping the object:
#-------------------------------------------------------------------------------
    
def dropped_on_changed ( object, name, event ):
    dropped = event.components[0]
    if isinstance( dropped, MyColorChip ):
        event.handled = True
        if isinstance( object, PlotValue ):
            if dropped._set_border:
                object.fill_color = dropped.fill_color
            else:
                object.line_color = dropped.fill_color
        else:    
            if dropped._set_border:
                object.border_color = dropped.fill_color
            else:
                object.bg_color = dropped.fill_color

#-------------------------------------------------------------------------------
#  Compute a noisy sine wave:
#-------------------------------------------------------------------------------

from Numeric     import pi, arange, sin
from RandomArray import random
                
def nsw ( ):    
    delta = pi / 30.0
    xx    = (arange( 0.0, 2.0 * pi + (delta / 2.0), delta ) * 1.5)
    return  (sin( xx ) + 0.4 * random( ( 61, ) ))

#-------------------------------------------------------------------------------
#  Enable test:
#-------------------------------------------------------------------------------

def main():
    pv3  = PlotValue( nsw(), line_weight = 2, plot_bg_color = 'clear' )
    pc2  = PlotComponent( pv3, bounds = ( 350, 10, 540, 150 ) )
    red_ball    = MyImage( image = 'red ball',   x = 250, y = 4 )
    blue_ball   = MyImage( image = 'blue ball',  x = 250, y = 120 )
    banana      = MyImage( image = 'banana',     x = 180, y = 300 )
    apple       = MyImage( image = 'apple',      x = 180, y = 450 )
    pear        = MyImage( image = 'pear',       x = 400, y = 300 )
    light_bulb  = MyImage( image = 'light bulb', x = 400, y = 450 )
    apple_shell = ImageFrame( MyImage( image = 'apple' ),
                              image = 'tan_imageshell',
                              x     = 650,
                              y     = 450 )
    profiler  = Profiler( image = 'profiler_off', x = 920, y = 10 )
    inspector = Inspector( x = 920, y = 50 )
    ititle    = draggable( ImageTitle( 'The Enable Toolkit Rocks!',
                                        bounds = ( 600, 380, 250, 40 ) ) ) 
    container = SimpleContainer(
       Box( fill_color = 'red',   bounds = (  130,   4, 100, 50 ) ), 
       Box( fill_color = 'blue',  bounds = (  130,  60, 100, 50 ) ), 
       Box( fill_color = 'green', bounds = (  130, 116, 100, 50 ) ),
       ResizeFrame( Box( fill_color = 'cyan' ),  
                    bounds = (  130, 172, 100, 50 ) ),
       WindowFrame( SimpleContainer( banana, apple, pear, light_bulb,
                                     fill_color = 'yellow', 
                                     bounds     = ( 130, 240, 400, 400 ) ),
                    title  = 'A Still Life',
                    bounds = ( 130, 240, 400, 400 ) ),
       WindowFrame( pc2, title        = 'A Chaco Plot',
                         image        = '=tan_window3',
                         bounds       = ( 350, 10, 540, 350 ),
                         color        = 'black',
                         shadow_color = 'white' ),
       red_ball, blue_ball, apple_shell, profiler, inspector,
       ititle,
       bounds = ( 0, 0, 960, 656 ) 
    )
    x = 4 
    y = 4
    for a in ( 1.0, 0.83, 0.67 ):                         
        for r in ( 0.0, 0.33, 0.67, 1.0 ):
            for g in ( 0.0, 0.33, 0.67, 1.0 ):
                for b in ( 0.0, 0.33, 0.67, 1.0 ):
                    container.add( MyColorChip( fill_color = ( r, g, b, a ),
                                                bounds = ( x, y, 16, 16 ) ) )
                    y += 20
                    if y >= 640:
                       y  = 4
                       x += 20
                       
    apple_shell.on_trait_change( left_down_changed_no_alpha, 'left_down'  )
    pv3.on_trait_change(         dropped_on_changed, 'dropped_on' )
    pv3.on_trait_change(         drag_enter_changed, 'drag_enter' )
    pv3.on_trait_change(         drag_leave_changed, 'drag_leave' )

    import Tkinter

    root = Tkinter.Tk()
    root.title( "Kiva Demo" )
    
    widget = Window( root, component = container )
    root.geometry( '960x656' )
    widget.pack( fill = "both", expand = 1 )

    root.mainloop()    


#-------------------------------------------------------------------------------
#  Program start-up:
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    main()            

