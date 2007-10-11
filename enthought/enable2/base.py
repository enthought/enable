"""
Define a base set of constants and functions used by the remainder of the
Enable package.
"""
#-------------------------------------------------------------------------------
#  Functions defined: bounding_box
#                     intersect_coordinates
#                     union_coordinates
#                     intersect_bounds
#                     union_bounds
#                     disjoint_intersect_coordinates
#                     does_disjoint_intersect_coordinates
#                     bounding_coordinates
#                     bounds_to_coordinates
#                     coordinates_to_bounds
#                     coordinates_to_size
#                     add_rectangles
#                     xy_in_bounds
#                     gc_image_for
#                     send_event_to
#                     subclasses_of
#-------------------------------------------------------------------------------

from __future__ import generators

# Major library imports
import sys
from os.path import dirname, splitext, abspath, join
from types import TypeType, TupleType
from zipfile import ZipFile, is_zipfile
from cStringIO import StringIO
from numpy import array, uint8

# Enthought library imports
from enthought.traits.api import TraitError

from enthought.kiva import GraphicsContext, font_metrics_provider
from enthought.kiva.backend_image import GraphicsContext as GraphicsContextArray
from enthought.kiva.constants import DEFAULT, DECORATIVE, ROMAN, SCRIPT, SWISS,\
                                     MODERN, NORMAL, BOLD, ITALIC
from enthought.kiva.fonttools import Font
from enthought.kiva.backend_image import Image, FontType

from colors import color_table, transparent_color

# Special 'empty rectangle' indicator:
empty_rectangle = -1

# Used to offset positions by half a pixel and bounding width/height by 1.
# TODO: Resolve this in a more intelligent manner.
half_pixel_bounds_inset = ( 0.5, 0.5, -1.0, -1.0 )

# Positions:
TOP          = 32
VCENTER      = 16
BOTTOM       =  8
LEFT         =  4
HCENTER      =  2
RIGHT        =  1

TOP_LEFT     = TOP    + LEFT
TOP_RIGHT    = TOP    + RIGHT
BOTTOM_LEFT  = BOTTOM + LEFT
BOTTOM_RIGHT = BOTTOM + RIGHT

# Text engraving style:
ENGRAVED = 1
EMBOSSED = 2
SHADOWED = 3

engraving_style = {
    'none':     0,
    'engraved': ENGRAVED,
    'embossed': EMBOSSED,
    'shadowed': SHADOWED
}

#-------------------------------------------------------------------------------
# Helper font functions
#-------------------------------------------------------------------------------

font_families = {
   'default':    DEFAULT,
   'decorative': DECORATIVE,
   'roman':      ROMAN,
   'script':     SCRIPT,
   'swiss':      SWISS,
   'modern':     MODERN
}
font_styles = {'italic': ITALIC}
font_weights = {'bold': BOLD}
font_noise = [ 'pt', 'point', 'family' ]

def str_to_font ( object, name, value ):
    "Converts a (somewhat) free-form string into a valid Font object."
    # FIXME: Make this less free-form and more well-defined.
    try:
        point_size = 10
        family     = SWISS
        style      = NORMAL
        weight     = NORMAL
        underline  = 0
        face_name  = []
        for word in value.split():
            lword = word.lower()
            if font_families.has_key( lword ):
               family = font_families[ lword ]
            elif font_styles.has_key( lword ):
               style = font_styles[ lword ]
            elif font_weights.has_key( lword ):
               weight = font_weights[ lword ]
            elif lword == 'underline':
               underline = 1
            elif lword not in font_noise:
               try:
                  point_size = int( lword )
               except:
                  face_name.append( word )
        return Font(face_name = " ".join(face_name),
                    size = point_size,
                    family = family,
                    weight = weight,
                    style = style,
                    underline = underline)
    except:
        pass
    raise TraitError, ( object, name, 'a font descriptor string',
                        repr( value ) )

str_to_font.info = ( "a string describing a font (e.g. '12 pt bold italic " +
                     "swiss family Arial' or 'default 12')" )

# Pick a default font that should work on all platforms.
default_font_name = 'modern 10'
default_font = str_to_font( None, None, default_font_name )

# A dummy graphics context just used to calculate font metrics:
#gc_temp = GraphicsContext( ( 1, 1 ) )
gc_temp = font_metrics_provider()

def filled_rectangle ( gc, position, bounds, bg_color = color_table["white"],
                       border_color = color_table["black"], border_size  = 1.0 ):
    "Draws a filled rectangle with border."
    gc.save_state()

    # Set up all the control variables for quick access:
    bsd = border_size + border_size
    bsh = border_size / 2.0
    x, y = position
    dx, dy = bounds

    # Fill the background region (if required):
    if bg_color is not transparent_color:
        gc.set_fill_color( bg_color )
        gc.begin_path()
        gc.rect( x + border_size, y + border_size, dx - bsd, dy - bsd )
        gc.fill_path()

    # Draw the border (if required):
    if border_size > 0:
        if border_color is not transparent_color:
            gc.set_stroke_color( border_color )
            gc.set_line_width( border_size )
            gc.begin_path()
            gc.rect( x + bsh, y + bsh, dx - border_size, dy - border_size )
            gc.stroke_path()

    gc.restore_state()
    return


# Image cache dictionary (indexed by 'normalized' filename):
_image_cache = {}
_zip_cache   = {}
_app_path    = None
_enable_path = None

def gc_image_for ( name, path = None ):
    "Convert an image file name to a cached Kiva gc containing the image"
    global _app_path, _enable_path
    filename = name
    if dirname( name ) == '':
        name = name.replace( ' ', '_' )
        if splitext( name )[1] == '':
            name += '.png'
        if path is None:
           if _enable_path is None:
              import enthought.enable2.base
              _enable_path = join( dirname( enthought.enable2.base.__file__ ),
                                   'images' )
           path = _enable_path
        elif path == '':
           if _app_path is None:
              _app_path = join( dirname( sys.argv[0] ), 'images' )
           path = _app_path
        else:
            if not isinstance(path, basestring):
                if not isinstance( path, TypeType ):
                    path = path.__class__
                path = join( dirname( sys.modules[ path.__module__ ].__file__ ),
                             'images' )
        filename = join( path, name.replace( ' ', '_' ).lower() )
    else:
        path = None
    filename = abspath( filename )
    image     = _image_cache.get( filename )
    if image is None:
        cachename = filename
        if path is not None:
            zip_path = abspath( path + '.zip' )
            zip_file = _zip_cache.get( zip_path )
            if zip_file is None:
               if is_zipfile( zip_path ):
                   zip_file = ZipFile( zip_path, 'r' )
               else:
                   zip_file = False
               _zip_cache[ zip_path ] = zip_file
            if isinstance( zip_file, ZipFile ):
                try:
                    filename = StringIO( zip_file.read( name ) )
                except:
                    pass
        try:
            _image_cache[ cachename ] = image = Image( filename )
        except:
            _image_cache[ filename ] = info = sys.exc_info()[:2]
            raise info[0], info[1]
    elif type( image ) is TupleType:
        raise image[0], image[1]
    return image

def bounding_box ( components ):
    "Compute the bounding box for a set of components"
    bxl, byb, bxr, byt = bounds_to_coordinates( components[0].bounds )
    for component in components[1:]:
        xl, yb, xr, yt = bounds_to_coordinates( component.bounds )
        bxl = min( bxl, xl )
        byb = min( byb, yb )
        bxr = max( bxr, xr )
        byt = max( byt, yt )
    return ( bxl, byb, bxr, byt )

def intersect_coordinates ( coordinates1, coordinates2 ):
    "Compute the intersection of two coordinate based rectangles"
    if (coordinates1 is empty_rectangle) or ( coordinates2 is empty_rectangle):
        return empty_rectangle
    xl1, yb1, xr1, yt1 = coordinates1
    xl2, yb2, xr2, yt2 = coordinates2
    xl = max( xl1, xl2 )
    yb = max( yb1, yb2 )
    xr = min( xr1, xr2 )
    yt = min( yt1, yt2 )
    if (xr > xl) and (yt > yb):
        return ( xl, yb, xr, yt )
    return empty_rectangle

def intersect_bounds ( bounds1, bounds2 ):
    "Compute the intersection of two bounds rectangles"
    if (bounds1 is empty_rectangle) or (bounds2 is empty_rectangle):
        return empty_rectangle

    intersection = intersect_coordinates(
                        bounds_to_coordinates( bounds1 ),
                        bounds_to_coordinates( bounds2 ) )
    if intersection is empty_rectangle:
        return empty_rectangle
    xl, yb, xr, yt = intersection
    return ( xl, yb, xr - xl, yt - yb )

def union_coordinates ( coordinates1, coordinates2 ):
    "Compute the union of two coordinate based rectangles"
    if coordinates1 is empty_rectangle:
        return coordinates2
    elif coordinates2 is empty_rectangle:
        return coordinates1
    xl1, yb1, xr1, yt1 = coordinates1
    xl2, yb2, xr2, yt2 = coordinates2
    return ( min( xl1, xl2 ), min( yb1, yb2 ),
             max( xr1, xr2 ), max( yt1, yt2 ) )

def union_bounds ( bounds1, bounds2 ):
    "Compute the union of two bounds rectangles"
    xl, yb, xr, yt = union_coordinates(
                        bounds_to_coordinates( bounds1 ),
                        bounds_to_coordinates( bounds2 ) )
    if xl is None:
        return empty_rectangle
    return ( xl, yb, xr - xl, yt - yb )

def disjoint_union_coordinates ( coordinates_list, coordinates ):
    """
    Return the disjoint union of an already disjoint list of rectangles and a
    new rectangle:

    Note: The 'infinite' area rectangle is indicated by 'None'. The coordinates
      list may be empty.
    """
    # If we already have an 'infinite' area, then we are done:
    if coordinates_list is None:
        return None

    result = []
    todo   = [ coordinates ]

    # Iterate over each item in the todo list:
    i = 0
    while i < len( todo ):
        xl1, yb1, xr1, yt1 = todo[i]
        j      = 0
        use_it = True

        # Iterate over each item in the original list of rectangles:
        while j < len( coordinates_list ):
            xl2, yb2, xr2, yt2 = coordinates_list[j]

            # Test for non-overlapping rectangles:
            if (xl1 >= xr2) or (xr1 <= xl2) or (yb1 >= yt2) or (yt1 <= yb2):
                j += 1
                continue

            # Test for rect 1 being wholly contained in rect 2:
            x1inx2 = ((xl1 >= xl2) and (xr1 <= xr2))
            y1iny2 = ((yb1 >= yb2) and (yt1 <= yt2))
            if x1inx2 and y1iny2:
                use_it = False
                break

            # Test for rect 2 being wholly contained in rect 1:
            x2inx1 = ((xl2 >= xl1) and (xr2 <= xr1))
            y2iny1 = ((yb2 >= yb1) and (yt2 <= yt1))
            if x2inx1 and y2iny1:
                del coordinates_list[j]
                continue

            # Test for rect 1 being within rect 2 along the x-axis:
            if x1inx2:
                if yb1 < yb2:
                    if yt1 > yt2:
                        todo.append( ( xl1, yt2, xr1, yt1 ) )
                    yt1 = yb2
                else:
                    yb1 = yt2
                j += 1
                continue

            # Test for rect 2 being within rect 1 along the x-axis:
            if x2inx1:
                if yb2 < yb1:
                    if yt2 > yt1:
                        coordinates_list.insert( j, ( xl2, yt1, xr2, yt2 ) )
                        j += 1
                    coordinates_list[j] = ( xl2, yb2, xr2, yb1 )
                else:
                    coordinates_list[j] = ( xl2, yt1, xr2, yt2 )
                j += 1
                continue

            # Test for rect 1 being within rect 2 along the y-axis:
            if y1iny2:
                if xl1 < xl2:
                    if xr1 > xr2:
                        todo.append( ( xr2, yb1, xr1, yt1 ) )
                    xr1 = xl2
                else:
                    xl1 = xr2
                j += 1
                continue

            # Test for rect 2 being within rect 1 along the y-axis:
            if y2iny1:
                if xl2 < xl1:
                    if xr2 > xr1:
                        coordinates_list.insert( j, ( xr1, yb2, xr2, yt2 ) )
                        j += 1
                    coordinates_list[j] = ( xl2, yb2, xl1, yt2 )
                else:
                    coordinates_list[j] = ( xr1, yb2, xr2, yt2 )
                j += 1
                continue

            # Handle a 'corner' overlap of rect 1 and rect 2:
            if xl1 < xl2:
                xl = xl1
                xr = xl2
            else:
                xl = xr2
                xr = xr1
            if yb1 < yb2:
                yb  = yb2
                yt  = yt1
                yt1 = yb2
            else:
                yb  = yb1
                yt  = yt2
                yb1 = yt2
            todo.append( ( xl, yb, xr, yt ) )
            j += 1

        # If there is anything left of rect 1 to use, add it to the result:
        if use_it:
            result.append( ( xl1, yb1, xr1, yt1 ) )

        # Advance to the next rectangle in the todo list:
        i += 1

    # Return whatever's left in the original list plus whatever made it to the
    # result:
    return coordinates_list + result


def disjoint_intersect_coordinates ( coordinates_list, coordinates ):
    """
    Return the disjoint intersection of an already disjoint list of rectangles
    and a new rectangle:

    Note: The 'infinite' area rectangle is indicated by 'None'. The coordinates
          list may be empty.
    """
    # If new rectangle is empty, the result is empty:
    if coordinates is empty_rectangle:
        return []

    # If we have an 'infinite' area, then return the new rectangle:
    if coordinates_list is None:
        return [ coordinates ]

    result             = []
    xl1, yb1, xr1, yt1 = coordinates

    # Intersect the new rectangle against each rectangle in the list:
    for xl2, yb2, xr2, yt2 in coordinates_list:
        xl = max( xl1, xl2 )
        yb = max( yb1, yb2 )
        xr = min( xr1, xr2 )
        yt = min( yt1, yt2 )
        if (xr > xl) and (yt > yb):
            rectangle = ( xl, yb, xr, yt )
            result.append( rectangle )
            if rectangle == coordinates:
                break
    return result

def does_disjoint_intersect_coordinates ( coordinates_list, coordinates ):
    "Return whether a rectangle intersects a disjoint set of rectangles anywhere"
    # If new rectangle is empty, the result is empty:
    if coordinates is empty_rectangle:
        return False

    # If we have an 'infinite' area, then return the new rectangle:
    if coordinates_list is None:
        return True

    # Intersect the new rectangle against each rectangle in the list until an
    # non_empty intersection is found:
    xl1, yb1, xr1, yt1 = coordinates
    for xl2, yb2, xr2, yt2 in coordinates_list:
        if ((min( xr1, xr2 ) > max( xl1, xl2 )) and
            (min( yt1, yt2 ) > max( yb1, yb2 ))):
            return True
    return False

def bounding_coordinates ( coordinates_list ):
    "Return the bounding rectangle for a list of rectangles"
    if coordinates_list is None:
        return None
    if len( coordinates_list ) == 0:
        return empty_rectangle
    xl, yb, xr, yt = 1.0E10, 1.0E10, -1.0E10, -1.0E10
    for xl1, yb1, xr1, yt1 in coordinates_list:
        xl = min( xl, xl1 )
        yb = min( yb, yb1 )
        xr = max( xr, xr1 )
        yt = max( yt, yt1 )
    return ( xl, yb, xr, yt )

def bounds_to_coordinates ( bounds ):
    "Convert a bounds rectangle to a coordinate rectangle"
    x, y, dx, dy = bounds
    return ( x, y, x + dx, y + dy )

def coordinates_to_bounds ( coordinates ):
    "Convert a coordinates rectangle to a bounds rectangle"
    xl, yb, xr, yt = coordinates
    return ( xl, yb, xr - xl, yt - yb )

def coordinates_to_size ( coordinates ):
    "Convert a coordinates rectangle to a size tuple"
    xl, yb, xr, yt = coordinates
    return ( xr - xl, yt - yb )

def add_rectangles ( rectangle1, rectangle2 ):
    "Add two bounds or coordinate rectangles"
    return ( rectangle1[0] + rectangle2[0],
             rectangle1[1] + rectangle2[1],
             rectangle1[2] + rectangle2[2],
             rectangle1[3] + rectangle2[3] )

def xy_in_bounds ( x, y, bounds ):
    "Test whether a specified (x,y) point is in a specified bounds"
    x0, y0, dx, dy = bounds
    return (x0 <= x < x0 + dx) and (y0 <= y < y0 + dy)

def send_event_to ( components, event_name, event ):
    "Send an event to a specified set of components until it is 'handled'"
    pre_event_name = 'pre_' + event_name
    for component in components:
        setattr( component, pre_event_name, event )
        if event.handled:
            return len( components )
    for i in xrange( len( components ) - 1, -1, -1 ):
        setattr( components[i], event_name, event )
        if event.handled:
            return i
    return 0

def subclasses_of ( klass ):
    "Generate all of the classes (and subclasses) for a specified class"
    yield klass
    for subclass in klass.__bases__:
        for result in subclasses_of( subclass ):
            yield result
    return

class IDroppedOnHandler:
    "Interface for draggable objects that handle the 'dropped_on' event"
    def was_dropped_on ( self, component, event ):
        raise NotImplementedError


# EOF
