"""
Define KeyBinding and KeyBindings classes which manages the mapping of Enable 
keystroke events into method calls on an application supplied controller object.
"""

from enthought.traits.api import Trait, TraitError, HasStrictTraits, Str, List, \
                             Any, Instance, Event
from enthought.traits.ui.api import View, Item, ListEditor
from enthought.traits.ui.wx.editor import Editor
from enthought.traits.ui.wx.editor_factory import EditorFactory

from enthought.util.wx.dialog import confirmation

from controls import Label
from wx_backend import Window
    

class KeyBindingEditorFactory(EditorFactory):

    def simple_editor ( self, ui, object, name, description, parent ):
        return SimpleKeyBindingEditor( parent,
                                       factory     = self, 
                                       ui          = ui, 
                                       object      = object, 
                                       name        = name, 
                                       description = description ) 
    
    def custom_editor ( self, ui, object, name, description, parent ):
        return SimpleKeyBindingEditor( parent,
                                       factory     = self, 
                                       ui          = ui, 
                                       object      = object, 
                                       name        = name, 
                                       description = description ) 
    
    def text_editor ( self, ui, object, name, description, parent ):
        return SimpleKeyBindingEditor( parent,
                                       factory     = self, 
                                       ui          = ui, 
                                       object      = object, 
                                       name        = name, 
                                       description = description ) 
    
    def readonly_editor ( self, ui, object, name, description, parent ):
        return SimpleKeyBindingEditor( parent,
                                       factory     = self, 
                                       ui          = ui, 
                                       object      = object, 
                                       name        = name, 
                                       description = description ) 

class SimpleKeyBindingEditor ( Editor ):
    
    def init ( self, parent ):
        """
        Finishes initializing the editor by creating the underlying toolkit
        widget.
        """
        self._binding = binding = Label( accepts_focus = True, 
                                         font          = 'Arial 9' )
        window        = Window( parent, component = binding )
        self.control  = window.control
        self.control.SetSize( ( 160, 22 ) )
        binding.on_trait_change( self.update_object, 'key' )
        binding.on_trait_change( self.update_focus,  'has_kbd_focus' )
        return

    def update_object ( self, event ):
        """
        Handles the user entering input data in the edit control.
        """
        try:
            self.value = value = key_event_to_name( event )
            self._binding.text = value
        except:
            pass
        return

    def update_editor ( self ):
        """
        Updates the editor when the object trait changes external to the 
        editor.
        """
        binding      = self._binding
        binding.text = self.str_value
        if binding._first is None:
            binding._first = False
            binding._text  = binding.text
        elif binding._text == binding.text:
            binding.bg_color = ( 1.0, 1.0, 1.0, 1.0 )
        else:
            binding.bg_color = ( 0.92, 0.92, 1.0, 1.0 )
        return
    
    def update_focus ( self, has_focus ):
        "Updates the current focus setting of the control"
        if has_focus:
            self._binding.border_size = 1
            self.object.owner.focus_owner = self._binding
        return

    def error ( self, excp ):
        """
        Handles an error that occurs while setting the object's trait value.
        """
        pass

# Create a default editor factory to use:        
key_binding_editor_factory = KeyBindingEditorFactory()  

# Key binding trait definition:
def validate_key_binding ( object, name, value ):
    if (object is None) or (object.owner is None):
        return value
    return object.owner.validate_key_binding( object, name, value )
    
Binding = Trait( '', validate_key_binding, event  = 'binding',
                                           editor = key_binding_editor_factory )
    

class KeyBinding ( HasStrictTraits ):
    
    # First key binding:
    binding1    = Binding
    
    # Second key binding:
    binding2    = Binding

    # Description of what application function the method performs:
    description = Str
    
    # Name of controller method the key is bound to:
    method_name = Str
    
    # KeyBindings object that 'owns' the KeyBinding:
    owner       = Instance( 'KeyBindings' )
    
    #---------------------------------------------------------------------------
    #  Traits view definitions:  
    #---------------------------------------------------------------------------
    
    traits_view = View( [ 'binding1', 'binding2', 'description~#', '-<>' ] )
    
    #---------------------------------------------------------------------------
    #  Handles a binding trait being changed:  
    #---------------------------------------------------------------------------
    
    def _binding_changed ( self ):
        if self.owner is not None:
            self.owner.binding_modified = self

#-------------------------------------------------------------------------------
#  'KeyBindings' class:  
#-------------------------------------------------------------------------------
                      
class KeyBindings ( HasStrictTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
        
    # Set of defined key bindings (added dynamically):
    #bindings = List( KeyBinding )
    
    # Optional prefix to add to each method name:
    prefix   = Str
    
    # Optional suffix to add to each method name:
    suffix   = Str
    
    # Event fired when one of the contained KeyBinding objects is changed:
    binding_modified = Event( KeyBinding )
    
    # Control that currently has the focus (if any):
    focus_owner      = Any
    
    traits_view = View( [ Item( 'bindings@#', 
                                editor = ListEditor( style = 'custom' ) ), 
                          '|{Click on a first or second column entry, then '
                          'press the key to assign to the corresponding '
                          'function}<>' ],
                        title     = 'Update Key Bindings',
                        kind      = 'livemodal',
                        resizable = True,
                        width     = 0.45,
                        height    = 0.50,
                        help      = False )

    def __init__ ( self, *bindings, **traits ):
        super( KeyBindings, self ).__init__( **traits )
        n = len( bindings )
        self.add_trait( 'bindings', List( KeyBinding, minlen = n, 
                                                      maxlen = n, 
                                                      mode   = 'list' ) )
        self.bindings = list( bindings )
        for binding in bindings:
            binding.owner = self
        return

    def do ( self, event, controller, *args ):
        "Processes a keyboard event."
        key_name = key_event_to_name( event )
        for binding in self.bindings:
            if (key_name == binding.binding1) or (key_name == binding.binding2):
                method_name = '%s%s%s' % ( 
                              self.prefix, binding.method_name, self.suffix )
                getattr( controller, method_name )( event, *args )
                break
        return

    def merge ( self, key_bindings ):
        "Merges another set of key bindings into this set."
        binding_dic = {}
        for binding in self.bindings:
            binding_dic[ binding.method_name ] = binding
            
        for binding in key_bindings.bindings:
            binding2 = binding_dic.get( binding.method_name )
            if binding2 is not None:
                binding2.binding1 = binding.binding1
                binding2.binding2 = binding.binding2
        return
        
    def validate_key_binding ( self, binding, name, key_name ):
        "Verifies that a specified key binding is valid."
        if key_name == '':
            return key_name
            
        for a_binding in self.bindings:
            if a_binding is not binding:
                if ((key_name == a_binding.binding1) or 
                    (key_name == a_binding.binding2)):
                    rc = confirmation( None, 
                             "'%s' has already been assigned to '%s'.\n"
                             "Do you wish to continue?" % ( 
                             key_name, a_binding.description ),
                             'Duplicate Key Definition' )
                    if rc == 5104:
                        raise TraitError
                        
        return key_name

    def _binding_modified_changed ( self, binding ):
        binding1 = binding.binding1
        binding2 = binding.binding2
        for a_binding in self.bindings:
            if binding is not a_binding:
                if binding1 == a_binding.binding1:
                    a_binding.binding1 = ''
                if binding1 == a_binding.binding2:
                    a_binding.binding2 = ''
                if binding2 == a_binding.binding1:
                    a_binding.binding1 = ''
                if binding2 == a_binding.binding2:
                    a_binding.binding2 = ''
        return
    
    def _focus_owner_changed ( self, old, new ):
        if old is not None:
            old.border_size = 0
        return

#-- object overrides -----------------------------------------------------------

    def __setstate__ ( self, state ):
        "Restores the state of a previously pickled object."
        n = len( state[ 'bindings' ] )
        self.add_trait( 'bindings', List( KeyBinding, minlen = n, maxlen = n ) )
        self.__dict__.update( state )
        self.bindings = self.bindings[:]
        return

def key_event_to_name ( event ):
    "Converts an Enable keystroke event into a corresponding key name."
    name = ''
    if event.alt_down:
        name = 'Alt'
        
    if event.control_down:
        name += '-Ctrl'
            
    c = event.character
    if event.shift_down and ((name != '') or (len( c ) > 1)):
        name += '-Shift'
        
    if c == ' ':
        c = 'Space'
    
    name += ('-' + c)
    
    if name[:1] == '-':
        return name[1:]
    return name

# EOF
