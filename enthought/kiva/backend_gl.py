
# Major library imports
import ctypes
from math import floor
from numpy import array, ndarray

# Pyglet and pyglet-related imports
from pyglet.font import Text
from pyglet.font import load as load_font
from pyglet.font.base import Font as PygletFont
from pyglet import gl
from pygarrayimage.arrayimage import ArrayInterfaceImage

# Local kiva imports
from affine import affine_from_values, transform_points
from agg import GraphicsContextGL as _GCL
from agg import GraphicsContextArray
from agg import AggFontType
from agg import Image
from agg import CompiledPath
from constants import BOLD, BOLD_ITALIC, ITALIC
from fonttools import Font


class ArrayImage(ArrayInterfaceImage):
    """ pyglet ImageData made from numpy arrays.

    Customized from pygarrayimage's ArrayInterfaceImage to override the texture
    creation.
    """

    def create_texture(self, cls):
        '''Create a texture containing this image.

        If the image's dimensions are not powers of 2, a TextureRegion of
        a larger Texture will be returned that matches the dimensions of this
        image.

        :Parameters:
            `cls` : class (subclass of Texture)
                Class to construct.

        :rtype: cls or cls.region_class
        '''

        texture = cls.create_for_size(
            gl.GL_TEXTURE_2D, self.width, self.height)
        subimage = False
        if texture.width != self.width or texture.height != self.height:
            texture = texture.get_region(0, 0, self.width, self.height)
            subimage = True

        internalformat = self._get_internalformat(self.format)

        gl.glBindTexture(texture.target, texture.id)
        gl.glTexParameteri(texture.target, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(texture.target, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(texture.target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(texture.target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)

        if subimage:
            width = texture.owner.width
            height = texture.owner.height
            blank = (ctypes.c_ubyte * (width * height * 4))()
            gl.glTexImage2D(texture.target, texture.level,
                         internalformat,
                         width, height,
                         1,
                         gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                         blank) 
            internalformat = None

        self.blit_to_texture(texture.target, texture.level, 
            0, 0, 0, internalformat)
        
        return texture 

    def blit_to_texture(self, target, level, x, y, z, internalformat=None):
        '''Draw this image to to the currently bound texture at `target`.

        If `internalformat` is specified, glTexImage is used to initialise
        the texture; otherwise, glTexSubImage is used to update a region.
        '''

        data_format = self.format
        data_pitch = abs(self._current_pitch)

        # Determine pixel format from format string
        matrix = None
        format, type = self._get_gl_format_and_type(data_format)
        if format is None:
            if (len(data_format) in (3, 4) and 
                gl.gl_info.have_extension('GL_ARB_imaging')):
                # Construct a color matrix to convert to GL_RGBA
                def component_column(component):
                    try:
                        pos = 'RGBA'.index(component)
                        return [0] * pos + [1] + [0] * (3 - pos)
                    except ValueError:
                        return [0, 0, 0, 0]
                # pad to avoid index exceptions
                lookup_format = data_format + 'XXX'
                matrix = (component_column(lookup_format[0]) +
                          component_column(lookup_format[1]) +
                          component_column(lookup_format[2]) + 
                          component_column(lookup_format[3]))
                format = {
                    3: gl.GL_RGB,
                    4: gl.GL_RGBA}.get(len(data_format))
                type = gl.GL_UNSIGNED_BYTE

                gl.glMatrixMode(gl.GL_COLOR)
                gl.glPushMatrix()
                gl.glLoadMatrixf((gl.GLfloat * 16)(*matrix))
            else:
                # Need to convert data to a standard form
                data_format = {
                    1: 'L',
                    2: 'LA',
                    3: 'RGB',
                    4: 'RGBA'}.get(len(data_format))
                format, type = self._get_gl_format_and_type(data_format)

        # Workaround: don't use GL_UNPACK_ROW_LENGTH
        if gl._current_context._workaround_unpack_row_length:
            data_pitch = self.width * len(data_format)

        # Get data in required format (hopefully will be the same format it's
        # already in, unless that's an obscure format, upside-down or the
        # driver is old).
        data = self._convert(data_format, data_pitch)

        if data_pitch & 0x1:
            alignment = 1
        elif data_pitch & 0x2:
            alignment = 2
        else:
            alignment = 4
        row_length = data_pitch / len(data_format)
        gl.glPushClientAttrib(gl.GL_CLIENT_PIXEL_STORE_BIT)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, alignment)
        gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, row_length)
        self._apply_region_unpack()
        gl.glTexParameteri(target, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(target, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        

        if target == gl.GL_TEXTURE_3D:
            assert not internalformat
            gl.glTexSubImage3D(target, level,
                            x, y, z,
                            self.width, self.height, 1,
                            format, type,
                            data)
        elif internalformat:
            gl.glTexImage2D(target, level,
                         internalformat,
                         self.width, self.height,
                         0,
                         format, type,
                         data)
        else:
            gl.glTexSubImage2D(target, level,
                            x, y,
                            self.width, self.height,
                            format, type,
                            data)
        gl.glPopClientAttrib()

        if matrix:
            gl.glPopMatrix()
            gl.glMatrixMode(gl.GL_MODELVIEW)
    

def image_as_array(img):
    """ Adapt an image object into a numpy array.

    Typically, this is used to adapt an agg GraphicsContextArray which has been
    used for image storage in Kiva applications.
    """
    if hasattr(img, 'bmp_array'):
        # Yup, a GraphicsContextArray.
        return img.bmp_array
    elif isinstance(img, ndarray):
        return img
    else:
        raise NotImplementedError("can't convert %r into a numpy array" % (img,))

def get_dpi():
    """ Returns the appropriate DPI setting for the system"""
    pass

class MRU(dict):
    def __init__(self, *args, **kw):
        # An ordering of keys in self; the last item was the most recently used
        self.__order__ = []
        self.__maxlength__ = 30
        dict.__init__(self, *args, **kw)

    def __getitem__(self, key):
        val = dict.__getitem__(self, key)
        # If we get here, then key was found in our dict
        self.__touch__(key)
        return val

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self.__touch__(key)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        if key in self.__order__:
            self.__order__.remove(key)

    def __touch__(self, key):
        """ Puts **key** as the most recently used item """
        if len(self.__order__) == 0:
            self.__order__.append(key)
        if (len(self.__order__) == self.__maxlength__) and key not in self.__order__:
            # The MRU is full, so pop the oldest element
            del self[self.__order__[0]]
        if key != self.__order__[-1]:
            try:
                ndx = self.__order__.index(key)
                self.__order__[ndx:-1] = self.__order__[ndx+1:]
                self.__order__[-1] = key
            except ValueError:
                # new key that's not already in the cache
                if len(self.__order__) == self.__maxlength__:
                    self.__order__ = self.__order__[1:] + [key]
                else:
                    self.__order__.append(key)
        return

# Use a singleton for the font cache
GlobalFontCache = MRU()


class GraphicsContext(_GCL):
    def __init__(self, size, *args, **kw):
        # Ignore the pix_format argument for now
        if "pix_format" in kw:
            kw.pop("pix_format")
        _GCL.__init__(self, size[0], size[1], *args, **kw)
        self.corner_pixel_origin = True
        self.pyglet_font = None

        # Maps Font instances to pyglet font instance
        self._font_cache = GlobalFontCache

    def set_font(self, font):
        """ **font** is either a kiva.agg.AggFontType, a kiva.fonttools.Font
            object, or a Pyglet Font object
        """
        # The font handling is a bit mangled because we are using
        # kiva.agg.graphics_state to store the font information on the state
        # stack.
        # In order to just use kiva.fonttools.Font, and circumvent using
        # kiva.agg.AggFontType, we will need to implement a parallel stack and
        # supplement the GraphicsContextArray implementations of save_state and
        # restore_state.
        # For now, though, we do this clumsy process of converting the AggFontType
        # to a kiva.fonttools.Font object.
        if isinstance(font, PygletFont):
            pyglet_font = font
        else:
            key = (font.name, font.size, font.family, font.style)
            if key not in self._font_cache:
                if isinstance(font, AggFontType):
                    agg_font = font
                    font = Font(face_name = agg_font.name,
                                size = agg_font.size,
                                family = agg_font.family,
                                style = agg_font.style)
                bold = False
                italic = False
                if font.style == BOLD or font.style == BOLD_ITALIC or font.weight == BOLD:
                    bold = True
                if font.style == ITALIC or font.style == BOLD_ITALIC:
                    italic = True
                pyglet_font = load_font(font.findfontname(), font.size, bold, italic)
                self._font_cache[key] = pyglet_font
            else:
                pyglet_font = self._font_cache[key]

        self.pyglet_font = pyglet_font
        return True

    def get_text_extent(self, text):
        if self.pyglet_font is None:
            return (0, 0, 0, 0)
        # See note in show_text_at_point about the valign argument.
        pyglet_text = Text(self.pyglet_font, text, valign=Text.BOTTOM)
        return (0, 0, pyglet_text.width, pyglet_text.height)

    def show_text(self, text, point = None):
        if point is None:
            point = (0,0)
        return self.show_text_at_point(text, *point)

    def show_text_at_point(self, text, x, y):
        # XXX: make this use self.get_font() to get the font from the state stack
        # rather than from the last font passed in to set_font()
        if self.pyglet_font is None:
            return False

        xform = self.get_ctm()
        x0 = xform[4]
        y0 = xform[5]

        # The GL backend places the center of a pixel at (0.5, 0.5); however, 
        # for show_text_at_point, we don't actually want to render the text
        # offset by half a pixel.  There is probably a better, more uniform way
        # to handle this across all of Kiva, because this is probably a common
        # issue that will arise, but for now, we just round the position down.
        x = floor(x + x0)
        y = floor(y + y0)
        
        # Use valign=Text.BOTTOM because by default, pyglet sets the baseline to
        # the y coordinate given.  Unfortunately, it doesn't expose a per-Text
        # descent (only a per-Font descent), so it's impossible to know how to 
        # offset the y value properly for a given string.  The only solution I've
        # found is to just set valign to BOTTOM.
        pyglet_text = Text(self.pyglet_font, text, x=x, y=y, valign=Text.BOTTOM)
        pyglet_text.color = self.get_fill_color()
        pyglet_text.draw()
        return True

    def draw_image(self, img, rect=None, force_copy=False):
        """ Renders a GraphicsContextArray into this GC """
        xform = self.get_ctm()
        x0 = xform[4]
        y0 = xform[5]

        image = image_as_array(img)
        shape = image.shape
        if shape[2] == 4:
            fmt = "RGBA"
        else:
            fmt = "RGB"
        aii = ArrayImage(image, format=fmt)
        texture = aii.texture

        # The texture coords consists of (u,v,r) for each corner of the
        # texture rectangle.  The coordinates are stored in the order
        # bottom left, bottom right, top right, top left.
        x, y, w, h = rect
        texture.width = w
        texture.height = h
        t = texture.tex_coords
        points = array([
            [x,   y+h],
            [x+w, y+h],
            [x+w, y],
            [x,   y],
        ])
        p = transform_points(affine_from_values(*xform), points)
        a = (gl.GLfloat*32)(
            t[0],   t[1],   t[2],  1.,
            p[0,0], p[0,1], 0,     1.,
            t[3],   t[4],   t[5],  1.,
            p[1,0], p[1,1], 0,     1.,
            t[6],   t[7],   t[8],  1.,
            p[2,0], p[2,1], 0,     1.,
            t[9],   t[10],  t[11], 1.,
            p[3,0], p[3,1], 0,     1.,
        )
        gl.glPushAttrib(gl.GL_ENABLE_BIT)
        gl.glEnable(texture.target)
        gl.glBindTexture(texture.target, texture.id)
        gl.glPushClientAttrib(gl.GL_CLIENT_VERTEX_ARRAY_BIT)
        gl.glInterleavedArrays(gl.GL_T4F_V4F, 0, a)
        gl.glDrawArrays(gl.GL_QUADS, 0, 4)
        gl.glPopClientAttrib()
        gl.glPopAttrib()

def font_metrics_provider():
    return GraphicsContext((1,1))

Canvas = None
CanvasWindow = None
