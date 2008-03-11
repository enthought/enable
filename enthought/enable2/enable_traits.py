"""
Define the base Enable object traits
"""

# Major library imports
from numpy import arange, array
from types import ListType, TupleType

# Enthought library imports
from enthought.kiva.traits.kiva_font_trait import KivaFont
from enthought.traits.api import HasTraits, Trait, TraitError, Range, Undefined,\
                             TraitPrefixList, TraitPrefixMap, TraitHandler, \
                             Delegate, Str, Float, List, CList, TraitFactory
from enthought.traits.ui.api import ImageEnumEditor, EnumEditor, FileEditor, TupleEditor, \
                                TextEditor, Handler

# Relative imports
import base
from base import default_font_name, engraving_style, gc_image_for

#------------------------------------------------------------------------------
#  Constants:
#------------------------------------------------------------------------------

# numpy 'array' type:
ArrayType = type( arange( 1.0 ) )

# Basic sequence types:
basic_sequence_types = ( ListType, TupleType )

# Sequence types:
sequence_types = [ ArrayType, ListType, TupleType ]

# Valid pointer shape names:
pointer_shapes = [
   'arrow', 'right arrow', 'blank', 'bullseye', 'char', 'cross', 'hand',
   'ibeam', 'left button', 'magnifier', 'middle button', 'no entry',
   'paint brush', 'pencil', 'point left', 'point right', 'question arrow',
   'right button', 'size top', 'size bottom', 'size left', 'size right',
   'size top right', 'size bottom left', 'size top left', 'size bottom right',
   'sizing', 'spray can', 'wait', 'watch', 'arrow wait'
]

# Cursor styles:
CURSOR_X = 1
CURSOR_Y = 2

cursor_styles = {
    'default':    -1,
    'none':       0,
    'horizontal': CURSOR_Y,
    'vertical':   CURSOR_X,
    'both':       CURSOR_X | CURSOR_Y
}

class TraitImage(TraitHandler):

    def __init__(self, allow_none = True):
        self.allow_none = allow_none
        return

    def validate(self, object, name, value):
        if self.allow_none and ((value is None) or (value == '')):
            setattr( object, '_' + name, None )
            return None
        path   = ''
        image  = value
        prefix = image[:1]
        if prefix == '=':
            path  = object
            image = image[1:]
        elif prefix == '.':
            path  = None
            image = image[1:]
        image_ = gc_image_for( image, path )
        if image_ is not None:
            setattr( object, '_' + name, image_ )
            return value
        self.error( object, name, self.repr( value ) )

    def info(self):
        return 'the name of an image file (e.g a .png, .jpg, .gif file)'

border_size_editor = ImageEnumEditor(
                         values = [ x for x in range( 9 ) ],
                         suffix = '_weight',
                         cols   = 3,
                         module = base )


#-------------------------------------------------------------------------------
# LineStyle trait
#-------------------------------------------------------------------------------

# Privates used for specification of line style trait.
__line_style_trait_values = {
    'solid':     None,
    'dot dash':  array( [ 3.0, 5.0, 9.0, 5.0 ] ),
    'dash':      array( [ 6.0, 6.0 ] ),
    'dot':       array( [ 2.0, 2.0 ] ),
    'long dash': array( [ 9.0, 5.0 ] )
}
__line_style_trait_map_keys = __line_style_trait_values.keys()
LineStyleEditor = EnumEditor( values=__line_style_trait_map_keys)

def __line_style_trait( value='solid', **metadata ):
    return Trait( value, __line_style_trait_values,
                    editor=LineStyleEditor, **metadata)

# A mapped trait for use in specification of line style attributes.
LineStyle = TraitFactory( __line_style_trait )


#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# Font trait:
font_trait = KivaFont(default_font_name)

# Bounds trait
bounds_trait = CList( [0.0, 0.0] )      # (w,h)
coordinate_trait = CList( [0.0, 0.0] )  # (x,y)

#bounds_trait = Trait((0.0, 0.0, 20.0, 20.0), valid_bounds, editor=bounds_editor)

# Component minimum size trait
# PZW: Make these just floats, or maybe remove them altogether.
ComponentMinSize = Range(0.0, 99999.0)
ComponentMaxSize = ComponentMinSize(99999.0)

# Pointer shape trait:
Pointer = Trait('arrow', TraitPrefixList(pointer_shapes))

# Cursor style trait:
cursor_style_trait = Trait('default', TraitPrefixMap(cursor_styles))

# Text engraving style:
engraving_trait = Trait ('none', TraitPrefixMap(engraving_style), cols = 4)

spacing_trait = Range(0, 63, value = 4)
padding_trait = Range(0, 63, value = 4)
margin_trait = Range(0, 63)
border_size_trait = Range(0,  8, editor = border_size_editor)

# Simple image trait:
image_trait = Trait(None, TraitImage(), editor = FileEditor)
string_image_trait = Str(editor = FileEditor)

# Time interval trait:
TimeInterval = Trait(None, None, Range(0.0, 3600.0))

# Stretch traits:
Stretch = Range(0.0, 1.0, value = 1.0)
NoStretch = Stretch(0.0)

# EOF
