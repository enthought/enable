#-------------------------------------------------------------------------------
#
#  The 'enable_test' plug-in for testing Enable components within the
#  enVisiable framework.
#
#  Written by: David C. Morrill
#
#  Date: 10/31/2003
#
#  (c) Copyright 2003 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

""" The Enable test plugin. """

import wx

from enthought.util.numerix import arange, sin, pi

from enthought.envisage.node            import Node
from enthought.envisage.node_tree_model import NodeTreeModel

from enthought.enable import FilledContainer, Inspector, ColorChip, \
                               ComponentFactory, LabelTraits, Label, Button, \
                               CheckBoxButton, RadioButton, CheckBox, Radio, \
                               ImageTitle, WindowFrame, TitleFrame, ImageFrame,\
                               Image, DraggableImage, Layout, Frame, ScrollBar,\
                               Scrolled, IDroppedOnHandler, GriddedCanvas, \
                               GuideLine, SelectionFrame, Splitter, Box, \
                               BaseContainer, Component

from enthought.enable.wx import Window
                               
from enthought.enable.enable_traits     import string_image_trait

from enthought.chaco.plot_component     import PlotComponent
from enthought.chaco.plot_value         import PlotValue

from enthought.envisage.explorer_view   import ExplorerView

from enthought.traits.api                   import Trait, HasTraits
from enthought.traits.ui.api                import Handler

from color_cycler                       import ColorCycler                                               

#-------------------------------------------------------------------------------
#  Global data:
#-------------------------------------------------------------------------------
               
application = None
                                           
#-------------------------------------------------------------------------------
#  'RootNode' class:
#-------------------------------------------------------------------------------

class RootNode ( Node ):
    
    #---------------------------------------------------------------------------
    #  Initialize the object: 
    #---------------------------------------------------------------------------
    
    def __init__ ( self, name, icon = None, **traits ):
        if icon is None:
            icon = 'images/templates.gif'
        Node.__init__( self, name       = name,
                             icon       = icon,
                             can_copy   = False,
                             can_rename = False,
                             can_cut    = False,
                             can_delete = False,
                             **traits )
        
#-------------------------------------------------------------------------------
#  'Template' class (implements the ComponentFactory interface):
#-------------------------------------------------------------------------------

class Template:
    
    #---------------------------------------------------------------------------
    #  Initialize the object: 
    #---------------------------------------------------------------------------
    
    def __init__ ( self, factory, 
                         args      = None, 
                         traits    = None, 
                         component = None ):
        self._factory = factory
        self._args    = args   or ()
        self._traits  = traits or {}

    #---------------------------------------------------------------------------
    #  Create an instance of the component that the factory produces: 
    #---------------------------------------------------------------------------
    
    def create_component ( self ):
        factory = self._factory
        if isinstance(factory, basestring):
            return eval( factory )
        return factory( *self._args, **self._traits )

#-------------------------------------------------------------------------------
#  'ComponentFactoryNode' class:
#-------------------------------------------------------------------------------

class ComponentFactoryNode ( Node ):
    """ A node representing an Enable component factory.
    """
    
    #---------------------------------------------------------------------------
    #  Initializes the object: 
    #---------------------------------------------------------------------------
    
    def __init__ ( self, template, **traits ):
        """ Creates a new node. """
        
        self.can_copy   = False
        self.can_cut    = False
        self.can_delete = False
        self.can_rename = False
        self.icon       = 'images/template.gif'

        # Base-class constructor:
        Node.__init__( self, data = template, **traits )

#-------------------------------------------------------------------------------
#  Create an appropriate component node for a specified Enable component:
#-------------------------------------------------------------------------------

def create_node_for ( component ):
    if isinstance( component, BaseContainer ):
        if hasattr( component, 'components' ):
            return ContainerNode( data = component )
        if hasattr( component, 'component' ):
            return SimpleContainerNode( data = component )
    return ComponentNode( data = component )

#-------------------------------------------------------------------------------
#  'ComponentNode' class:  
#-------------------------------------------------------------------------------

class ComponentNode ( Node ):
    """ A node representing an Enable component. """

    def __init__ ( self, **traits ):
        """ Create a new Enable component node. 
        """
        Node.__init__( self, **traits )
        self.name = self.data.__class__.__name__
        self.icon = 'images/component.gif'

    def property_sheet ( self, parent ):
        return None
        # ui_hack
        #obj    = self.data
        #traits = obj.editable_traits()
        #try:
        #    wrap = not isinstance( traits[0], TraitGroup )
        #except:
        #    wrap = True
        #if wrap:
        #    traits = TraitGroup( show_border = False, 
        #                         style       = 'simple', 
        #                         *traits )
        #return TraitSheet( parent, obj, traits, trait_sheet_handler )

#-------------------------------------------------------------------------------
#  'ContainerNode' class:  
#-------------------------------------------------------------------------------

class ContainerNode ( ComponentNode ):
    """ A node representing an Enable container. 
    """

    def __init__ ( self, **traits ):
        """ Create a new Enable container node. 
        """
        ComponentNode.__init__( self, **traits )
        self.icon = 'images/container.gif'
        self.data.on_trait_change( self.on_check_components, 
                                   'components_items' )
        self.check_components()                                  
            
    def on_check_components ( self ):
        if self.check_components() > 0:
            application.components.structure_changed( self )
        
    def check_components ( self ):
        components = self.data.components[:]
        count      = 0
        for child in self.children[:]:
            try:
                components.remove( child.data )
            except:
                self.remove( child )
                count += 1
        for component in components:
            self.append( create_node_for( component ) )
            count += 1
        return count

#-------------------------------------------------------------------------------
#  'SimpleContainerNode' class:  
#-------------------------------------------------------------------------------

class SimpleContainerNode ( ComponentNode ):
    """ A node representing a single component Enable container. 
    """

    def __init__ ( self, **traits ):
        """ Create a new Enable container node. 
        """
        ComponentNode.__init__( self, **traits )
        self.icon = 'images/simple_container.gif'
        self.data.on_trait_change( self.on_check_component, 'component' )
        self.check_component()                                  
            
    def on_check_component ( self ):
        self.check_component()
        application.components.structure_changed( self )
        
    def check_component ( self ):
        for child in self.children[:]:
            self.remove( child )
        self.append( create_node_for( self.data.component ) )
        
#-------------------------------------------------------------------------------
#  'MyContainer' class:  
#-------------------------------------------------------------------------------

class MyContainer ( FilledContainer ):
        
    def drag_over_by_componentfactorynode ( self, node, event ):
        event.handled = True

    def drag_leave_by_componentfactorynode ( self, node, event ):
        event.handled = True

    def dropped_on_by_componentfactorynode ( self, node, event ):
       try:
        event.handled = True
        template      = node.data
        component     = template.create_component()
        component.location( event.x, event.y )
        self.add( component )
       except:
           import traceback
           traceback.print_exc()
           
#-------------------------------------------------------------------------------
#  Create a gridded canvas window:
#-------------------------------------------------------------------------------
           
def gridded_canvas_window ( ):
    gc = GriddedCanvas( min_width = 2000, min_height = 2000 )
    gc.on_trait_change( edit_component, 'component_context' )
    return WindowFrame( Scrolled( gc ), 
                        width      = 400, height = 400, 
                        title      = 'GriddedCanvas' )
                        
def edit_component ( component ):
    component.edit_traits( handler = trait_sheet_handler )
        
#-------------------------------------------------------------------------------
#  Create the Enable component templates:
#-------------------------------------------------------------------------------
        
def create_templates ( root_node ):
    add = root_node.append
    cfn = ComponentFactoryNode
    add( cfn( Template( lambda: Label( 'Label' ) ),
              name = 'Label' ) )
    add( cfn( Template( lambda: CheckBox( 'CheckBox' ) ),
              name = 'Check Box' ) )
    add( cfn( Template( lambda: Radio( 'Radio' ) ),
              name = 'Radio' ) )
    add( cfn( Template( lambda: Button( 'Button' ) ),
              name = 'Button' ) )
    add( cfn( Template( lambda: CheckBoxButton( 'CheckBoxButton' ) ),
              name = 'Check Box Button' ) )
    add( cfn( Template( lambda: RadioButton( 'RadioButton' ) ),
              name = 'Radio Button' ) )
    add( cfn( Template( lambda: ImageTitle( 'ImageTitle' ) ),
              name = 'Image Title' ) )
    add( cfn( Template( lambda: WindowFrame( Label( 'Label' ), 
                                             title = 'WindowFrame' ) ),
              name = 'Window Frame' ) )
    add( cfn( Template( lambda: ScrollBar( height = 400 ) ), 
              name = 'Vertical Scrollbar' ) )
    add( cfn( Template( lambda: ScrollBar( style  = 'horizontal',
                                                    width = 400 ) ), 
              name = 'Horizontal Scrollbar' ) )
    add( cfn( Template( gridded_canvas_window ),
              name = 'Gridded Canvas' ) )
    add( cfn( Template( lambda: WindowFrame( Splitter(
                                           Label( "Label1", border_size = 1 ),
                                           Label( "Label2", border_size = 1 ) ), 
                                           width = 400, height = 400, 
                                           title = 'Splitter' ) ),
              name = 'Splitter' ) )
    add( cfn( Template( lambda: Layout( width = 200, height = 200 ) ),
              name = 'Layout' ) )
    add( cfn( Template( lambda: Image() ),
              name = 'Image' ) )
    add( cfn( Template( lambda: PlotWidget( width  = 200, height = 200 ) ),
              name = 'Plot Value' ) )
    add( cfn( Template( lambda: ColorCycler( width = 75, height = 75 ) ),
              name = 'Color Cycler' ) )
        
#-------------------------------------------------------------------------------
#  Create the simple 'Enable' window:
#-------------------------------------------------------------------------------
        
def create_enable_window ( app, frame, location ):
    """ Creates a simple Enable window. """
    
    global application
    application = app
    
    app.container = container = MyContainer( 
                                         bg_color = ( .925, .914, .847, 1.0 ),
                                         bounds = ( 0.0, 0.0, 2000.0, 2000.0 ) )
    ew = Window( location, -1, component = container )
    location.AddPage( ew, 'Simple Canvas' )
    
    base = 200                            
    container.add( Inspector( x = 5, y = base - 130 ),
                   ColorChip( item = 'fg_color',     x = 5, y = base ),
                   ColorChip( item = 'bg_color',     x = 5, y = base - 32 ),
                   ColorChip( item = 'shadow_color', x = 5, y = base - 64 ),
                   ColorChip( item = 'alt_color',    x = 5, y = base - 96 ), 
                   Remover( x = 5, y = base - 164 ), 
                   Bouncer( x = 5, y = base - 198 ) 
    )
    
    # Add some handy references to the PyCrust shell name space:
    bind = frame.shell.bind
    bind( 'container', container )
    bind( 'cn',        container )
    bind( 'gc_stats',  gc_stats )
    for cls in [ FilledContainer, Window, Inspector, ColorChip, Component,
                 ComponentFactory, Label, Button, CheckBoxButton, RadioButton,
                 CheckBox, Radio, Splitter, ImageTitle, WindowFrame, TitleFrame, 
                 ImageFrame, Image, ScrollBar, Scrolled, Layout, LabelTraits, 
                 GriddedCanvas, GuideLine, SelectionFrame, BouncingBall, Box ]:
        bind( cls.__name__, cls ) 
        
    # Mark all the initial objects as permanent (i.e. non-deletable):
    for component in container.components:
        component._is_permanent = True
        
    return container
        
#-------------------------------------------------------------------------------
#  Create the gridded canvas 'Enable' window:
#-------------------------------------------------------------------------------
        
def create_gridded_canvas_window ( app, frame, location ):
    """ Creates a gridded canvas Enable window. """
    
    app.grid = grid = GriddedCanvas( width = 400, height = 400 ) 
    ew       = Window( location, -1, component = grid )
    location.AddPage( ew, 'Gridded Canvas' )
    
    # Add some handy references to the PyCrust shell name space:
    bind = frame.shell.bind
    bind( 'grid', grid )
        
    return grid
    
#-------------------------------------------------------------------------------
#  Create the 'active components' window:
#-------------------------------------------------------------------------------
       
def create_active_components ( app, frame, location ):
    """ Creates the 'active components' window.
    """
    panel = wx.Panel( location, -1 )
    sizer = wx.BoxSizer( wx.VERTICAL )
    panel.SetAutoLayout( True )
    panel.SetSizer( sizer )
    
    node = RootNode( name = 'Components' )
    node.append( ContainerNode( data = app.container ) )
    node.append( ContainerNode( data = app.grid ) )
    app.components = components = NodeTreeModel( node )
    components.structure_changed( node )
    components = ExplorerView( panel, -1, app, frame, model = components )
    sizer.Add( components, 1, wx.EXPAND )
    sizer.Fit( panel )
    location.AddPage( panel, 'Components' )
    return components
    
#-------------------------------------------------------------------------------
#  Create a PlotValue object:
#-------------------------------------------------------------------------------
    
def PlotWidget ( **traits ):
    x_values = arange( 0, 10 * pi + 0.001, pi / 50.0 )
    y_values = sin( x_values ) * x_values
    return PlotComponent( PlotValue( y_values, index = PlotValue( x_values ) ),
                          **traits )
                          
#-------------------------------------------------------------------------------
#  'Remover' class:
#-------------------------------------------------------------------------------
    
class Remover ( DraggableImage, IDroppedOnHandler ):    
    
    __traits__ = {
        'image':  Trait( '=hammer', string_image_trait )
    }
        
    def was_dropped_on ( self, component, event ):
        if ((not component._is_permanent) and 
            (component.container is self.container)):
            component.container.remove( component )
    
#-------------------------------------------------------------------------------
#  'BouncingBall' class:  
#-------------------------------------------------------------------------------

class BouncingBall ( Image ):
    
    __traits__ = {
        'image':  Trait( '=red_ball', string_image_trait )
    }
    
    def left_down_changed ( self ):
        if self.timer_interval is None:
            if self._dx is None:
                self._dx = 10
                self._dy = 10
            self.timer_interval = 0.03
        else:
            self.timer_interval = None
            
    def timer_changed ( self ):
        try:
            dx, dy = self.dimensions()
            cn     = self.container
            x      = self.x + self._dx
            y      = self.y + self._dy
            if x < cn.left:
                self._dx = -self._dx
                x = 2.0 * cn.left - x
            elif x >= (cn.right - dx):
                self._dx = -self._dx
                x = 2 * (cn.right - dx) - x
            if y < cn.bottom:
                self._dy = -self._dy
                y = 2.0 * cn.bottom - y
            elif y >= (cn.top - dy):
                self._dy = -self._dy
                y = 2 * (cn.top - dy) - y
            self.location( x, y )
        except:
            self.timer_interval = None
    
#-------------------------------------------------------------------------------
#  'Bouncer' class:
#-------------------------------------------------------------------------------
    
class Bouncer ( DraggableImage, IDroppedOnHandler ):    

    image = Trait('=bouncer', string_iamge_strait)
        
    def was_dropped_on ( self, component, event ):
        if ((not component._is_permanent) and 
            (component.container is self.container)):
            component.container.add( BounceFrame( component ) )
        
#-------------------------------------------------------------------------------
#  'BounceFrame' class:
#-------------------------------------------------------------------------------

class BounceFrame ( Frame ):

    #---------------------------------------------------------------------------
    #  Initialize the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, component = None, **traits ):
        Frame.__init__( self, component, **traits )
        self._dx = self._dy = 10.0
        self.timer_interval = 0.03
        
    #---------------------------------------------------------------------------
    #  Generate any additional components that contain a specified (x,y) point:
    #---------------------------------------------------------------------------
       
    def _components_at ( self, x, y ):
        if len( self.component.components_at( x, y ) ) >= 0:
            return [ self ]
        return []
        
    #---------------------------------------------------------------------------
    #  Handle the user terminating the bouncer:
    #---------------------------------------------------------------------------
    
    def left_down_changed ( self ):
        self.timer_interval = None
        container = self.container
        container.remove( self )
        container.add( self.component )
    
    #---------------------------------------------------------------------------
    #  Handle the timer pop event:
    #---------------------------------------------------------------------------
    
    def timer_changed ( self ):
        try:
            dx, dy = self.dimensions()
            cn     = self.container
            x      = self.x + self._dx
            y      = self.y + self._dy
            if x < cn.left:
                self._dx = -self._dx
                x = 2.0 * cn.left - x
            elif x >= (cn.right - dx):
                self._dx = -self._dx
                x = 2 * (cn.right - dx) - x
            if y < cn.bottom:
                self._dy = -self._dy
                y = 2.0 * cn.bottom - y
            elif y >= (cn.top - dy):
                self._dy = -self._dy
                y = 2 * (cn.top - dy) - y
            self.location( x, y )
        except:
            self.timer_interval = None
            
#-------------------------------------------------------------------------------
#  Display garbage collector statistics:  
#-------------------------------------------------------------------------------

def gc_stats ( ):
    import gc
    klasses = {}
    for obj in gc.get_objects():
        try:
            klass = obj.__class__.__name__
        except:
            klass = '<unknown>'
        klasses[ klass ] = klasses.setdefault( klass, 0 ) + 1
    keys = klasses.keys()
    keys.sort( lambda x, y: cmp( klasses[y], klasses[x] ) )
    for key in keys:
        print '%7d: %s' % ( klasses[ key ], key )
