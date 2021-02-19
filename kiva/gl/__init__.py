# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from __future__ import absolute_import, print_function

import ctypes
from math import floor
import sys

from numpy import array, ndarray

# Local kiva imports
from kiva.affine import affine_from_values, transform_points
from kiva.constants import BOLD, BOLD_ITALIC, ITALIC
from kiva.fonttools import Font
from kiva.gl.gl import CompiledPath, GraphicsContextGL, KivaGLFontType


def image_as_array(img):
    """ Adapt an image object into a numpy array.

    Typically, this is used to adapt an agg GraphicsContextArray which has been
    used for image storage in Kiva applications.
    """
    from PIL import Image

    if hasattr(img, "bmp_array"):
        # Yup, a GraphicsContextArray.
        img = Image.fromarray(img.bmp_array)
    elif isinstance(img, ndarray):
        img = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        pass
    else:
        msg = "can't convert %r into a numpy array" % (img,)
        raise NotImplementedError(msg)

    # Ensure RGB or RGBA formats
    if not img.mode.startswith("RGB"):
        img = img.convert("RGB")
    return array(img)


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
        if (len(self.__order__) == self.__maxlength__ and
                key not in self.__order__):
            # The MRU is full, so pop the oldest element
            del self[self.__order__[0]]
        if key != self.__order__[-1]:
            try:
                ndx = self.__order__.index(key)
                self.__order__[ndx:-1] = self.__order__[ndx + 1:]
                self.__order__[-1] = key
            except ValueError:
                # new key that's not already in the cache
                if len(self.__order__) == self.__maxlength__:
                    self.__order__ = self.__order__[1:] + [key]
                else:
                    self.__order__.append(key)


# Pyglet and pyglet-related imports
# Before we import anything else from pyglet, we need to set the shadow_window
# option to False, so that it does not conflict with WX, in case someone is
# trying to use the kiva GL GraphicsContext from within WX.
# This is necessary as of pyglet 1.1.
try:
    import pyglet

    pyglet.options["shadow_window"] = False

    from pyglet.text import Label
    from pyglet.font import load as load_font
    from pyglet.font.base import Font as PygletFont
    from pyglet import gl
    from pygarrayimage.arrayimage import ArrayInterfaceImage

    class _ObjectSpace(object):
        """ Object space mocker

        Source: https://github.com/ColinDuquesnoy/QPygletWidget
        """

        def __init__(self):
            # Textures and buffers scheduled for deletion the next time this
            # object space is active.
            self._doomed_textures = []
            self._doomed_buffers = []

    class FakePygletContext(object):
        """ pyglet.gl.Context mocker.

        This is used to make pyglet believe that a valid context has already
        been setup. (Qt takes care of creating the open gl context)

        _Most of the methods are empty, there is just the minimum required to
        make it look like a duck..._

        Source: https://github.com/ColinDuquesnoy/QPygletWidget
        """

        # define the same class attribute as pyglet.gl.Context
        CONTEXT_SHARE_NONE = None
        CONTEXT_SHARE_EXISTING = 1
        _gl_begin = False
        _info = None
        _workaround_checks = [
            (
                "_workaround_unpack_row_length",
                lambda info: info.get_renderer() == "GDI Generic",
            ),
            (
                "_workaround_vbo",
                lambda info: info.get_renderer().startswith("ATI Radeon X"),
            ),
            (
                "_workaround_vbo_finish",
                lambda info: (
                    "ATI" in info.get_renderer()
                    and info.have_version(1, 5)
                    and sys.platform == "darwin"
                ),
            ),
        ]

        def __init__(self, context_share=None):
            """ Setup workaround attr and object spaces
                (again to mock what is done in pyglet context)
            """
            self.object_space = _ObjectSpace()
            for attr, check in self._workaround_checks:
                setattr(self, attr, None)

        def __repr__(self):
            return "%s()" % self.__class__.__name__

        def set_current(self):
            gl.current_context = self

        def destroy(self):
            pass

        def delete_texture(self, texture_id):
            pass

        def delete_buffer(self, buffer_id):
            pass

    class ArrayImage(ArrayInterfaceImage):
        """ pyglet ImageData made from numpy arrays.

        Customized from pygarrayimage's ArrayInterfaceImage to override the
        texture creation.
        """

        def create_texture(self, cls, rectangle=False, force_rectangle=False):
            """ Create a texture containing this image.

            If the image's dimensions are not powers of 2, a TextureRegion of
            a larger Texture will be returned that matches the dimensions of
            this image.

            :Parameters:
                `cls` : class (subclass of Texture)
                    Class to construct.
                `rectangle` : bool
                    ``True`` if a rectangle can be created; see
                    `AbstractImage.get_texture`.

            :rtype: cls or cls.region_class
            """

            _is_pow2 = (lambda v: (v & (v - 1)) == 0)

            target = gl.GL_TEXTURE_2D
            if (rectangle
                    and not (_is_pow2(self.width) and _is_pow2(self.height))):
                if gl.gl_info.have_extension("GL_ARB_texture_rectangle"):
                    target = gl.GL_TEXTURE_RECTANGLE_ARB
                elif gl.gl_info.have_extension("GL_NV_texture_rectangle"):
                    target = gl.GL_TEXTURE_RECTANGLE_NV

            texture = cls.create_for_size(target, self.width, self.height)
            subimage = False
            if texture.width != self.width or texture.height != self.height:
                texture = texture.get_region(0, 0, self.width, self.height)
                subimage = True

            internalformat = self._get_internalformat(self.format)

            gl.glBindTexture(texture.target, texture.id)
            gl.glTexParameteri(
                texture.target, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE
            )
            gl.glTexParameteri(
                texture.target, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE
            )
            gl.glTexParameteri(
                texture.target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR
            )
            gl.glTexParameteri(
                texture.target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR
            )

            if subimage:
                width = texture.owner.width
                height = texture.owner.height
                blank = (ctypes.c_ubyte * (width * height * 4))()
                gl.glTexImage2D(
                    texture.target,
                    texture.level,
                    internalformat,
                    width,
                    height,
                    1,
                    gl.GL_RGBA,
                    gl.GL_UNSIGNED_BYTE,
                    blank,
                )
                internalformat = None

            self.blit_to_texture(
                texture.target, texture.level, 0, 0, 0, internalformat
            )

            return texture

        def blit_to_texture(self, target, level, x, y, z, internalformat=None):
            """Draw this image to to the currently bound texture at `target`.

            If `internalformat` is specified, glTexImage is used to initialise
            the texture; otherwise, glTexSubImage is used to update a region.
            """

            data_format = self.format
            data_pitch = abs(self._current_pitch)

            # Determine pixel format from format string
            matrix = None
            format, type = self._get_gl_format_and_type(data_format)
            if format is None:
                if (len(data_format) in (3, 4)
                        and gl.gl_info.have_extension("GL_ARB_imaging")):

                    # Construct a color matrix to convert to GL_RGBA
                    def component_column(component):
                        try:
                            pos = "RGBA".index(component)
                            return [0] * pos + [1] + [0] * (3 - pos)
                        except ValueError:
                            return [0, 0, 0, 0]

                    # pad to avoid index exceptions
                    lookup_format = data_format + "XXX"
                    matrix = (
                        component_column(lookup_format[0])
                        + component_column(lookup_format[1])
                        + component_column(lookup_format[2])
                        + component_column(lookup_format[3])
                    )
                    format = {3: gl.GL_RGB, 4: gl.GL_RGBA}.get(
                        len(data_format)
                    )
                    type = gl.GL_UNSIGNED_BYTE

                    gl.glMatrixMode(gl.GL_COLOR)
                    gl.glPushMatrix()
                    gl.glLoadMatrixf((gl.GLfloat * 16)(*matrix))
                else:
                    # Need to convert data to a standard form
                    data_format = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}.get(
                        len(data_format)
                    )
                    format, type = self._get_gl_format_and_type(data_format)

            # Workaround: don't use GL_UNPACK_ROW_LENGTH
            if gl.current_context._workaround_unpack_row_length:
                data_pitch = self.width * len(data_format)

            # Get data in required format (hopefully will be the same format
            # it's already in, unless that's an obscure format, upside-down or
            # the driver is old).
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
            gl.glTexParameteri(
                target, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE
            )
            gl.glTexParameteri(
                target, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE
            )
            gl.glTexParameteri(target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)

            if target == gl.GL_TEXTURE_3D:
                assert not internalformat
                gl.glTexSubImage3D(
                    target,
                    level,
                    x,
                    y,
                    z,
                    self.width,
                    self.height,
                    1,
                    format,
                    type,
                    data,
                )
            elif internalformat:
                gl.glTexImage2D(
                    target,
                    level,
                    internalformat,
                    self.width,
                    self.height,
                    0,
                    format,
                    type,
                    data,
                )
            else:
                gl.glTexSubImage2D(
                    target,
                    level,
                    x,
                    y,
                    self.width,
                    self.height,
                    format,
                    type,
                    data,
                )
            gl.glPopClientAttrib()

            if matrix:
                gl.glPopMatrix()
                gl.glMatrixMode(gl.GL_MODELVIEW)

    # Use a singleton for the font cache
    GlobalFontCache = MRU()

    def GetFont(font):
        """ Returns a Pylget Font object for the given Agg or Kiva font """
        if isinstance(font, PygletFont):
            pyglet_font = font
        else:
            # KivaGLFontType
            key = (font.name, font.size, font.family, font.style)
            if key not in GlobalFontCache:
                if isinstance(font, KivaGLFontType):
                    kiva_gl_font = font
                    font = Font(
                        face_name=kiva_gl_font.name,
                        size=kiva_gl_font.size,
                        family=kiva_gl_font.family,
                        style=kiva_gl_font.style,
                    )
                bold = False
                italic = False
                if font.style in [BOLD, BOLD_ITALIC] or font.weight == BOLD:
                    bold = True
                if font.style in [ITALIC, BOLD_ITALIC]:
                    italic = True
                pyglet_font = load_font(
                    font.findfontname(), font.size, bold, italic
                )
                GlobalFontCache[key] = pyglet_font
            else:
                pyglet_font = GlobalFontCache[key]
        return pyglet_font

    # Because Pyglet 1.1 uses persistent Label objects to efficiently lay
    # out and render text, we cache these globally to minimize the creation
    # time.  An MRU is not really the right structure to use here, though.
    # (We typically expect that the same numbers of labels will be rendered.)
    GlobalTextCache = MRU()
    GlobalTextCache.__maxlength__ = 100

    def GetLabel(text, pyglet_font):
        """ Returns a Pyglet Label object for the given text and font """
        key = (text, pyglet_font)
        if key not in GlobalTextCache:
            # Use anchor_y="bottom" because by default, pyglet sets the
            # baseline to the y coordinate given.  Unfortunately, it doesn't
            # expose a per-Text descent (only a per-Font descent), so it's
            # impossible to know how to offset the y value properly for a
            # given string.
            label = Label(
                text,
                font_name=pyglet_font.name,
                font_size=pyglet_font.size,
                anchor_y="bottom",
            )
            GlobalTextCache[key] = label
        else:
            label = GlobalTextCache[key]
        return label


except ImportError:
    # Pyglet is not available, so we forgo some features
    ArrayImage = None
    GetFont = None
    GetLabel = None
    gl = None


class GraphicsContext(GraphicsContextGL):
    def __init__(self, size, *args, **kw):
        # Ignore the pix_format argument for now
        kw.pop("pix_format", None)
        base_scale = kw.pop("base_pixel_scale", 1)
        GraphicsContextGL.__init__(self, size[0], size[1], *args, **kw)
        self.corner_pixel_origin = True

        # For HiDPI support
        self.scale_ctm(base_scale, base_scale)

        self._font_stack = []
        self._current_font = None
        self._text_pos = (0, 0)

    def save_state(self):
        super(GraphicsContext, self).save_state()
        self._font_stack.append(self._current_font)

    def restore_state(self):
        super(GraphicsContext, self).restore_state()
        self._current_font = self._font_stack.pop()

    def set_font(self, font):
        self._current_font = font

    def get_text_extent(self, text):
        if self._current_font is None:
            return (0, 0, 0, 0)

        pyglet_font = GetFont(self._current_font)
        label = GetLabel(text, pyglet_font)
        return (0, 0, label.content_width, label.content_height)

    def set_text_position(self, x, y):
        self._text_pos = (x, y)

    def show_text(self, text, point=None):
        if point is None:
            point = self._text_pos
        return self.show_text_at_point(text, *point)

    def show_text_at_point(self, text, x, y):
        if self._current_font is None:
            return

        pyglet_font = GetFont(self._current_font)
        label = GetLabel(text, pyglet_font)

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

        label.x = x
        label.y = y
        c = self.get_fill_color()
        label.color = (
            int(c[0] * 255),
            int(c[1] * 255),
            int(c[2] * 255),
            int(c[3] * 255),
        )
        label.draw()
        return True

    def linear_gradient(self, x1, y1, x2, y2, stops, spread_method,
                        units="userSpaceOnUse"):
        """ Not implemented.
        """
        pass

    def radial_gradient(self, cx, cy, r, fx, fy, stops, spread_method,
                        units="userSpaceOnUse"):
        """ Not implemented.
        """
        pass

    def draw_image(self, img, rect=None, force_copy=False):
        """ Renders a GraphicsContextArray into this GC """
        xform = self.get_ctm()

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
        points = array([[x, y + h], [x + w, y + h], [x + w, y], [x, y]])
        p = transform_points(affine_from_values(*xform), points)
        a = (gl.GLfloat*32)(
            t[0],    t[1],    t[2],  1.,
            p[0, 0], p[0, 1], 0,     1.,
            t[3],    t[4],    t[5],  1.,
            p[1, 0], p[1, 1], 0,     1.,
            t[6],    t[7],    t[8],  1.,
            p[2, 0], p[2, 1], 0,     1.,
            t[9],    t[10],   t[11], 1.,
            p[3, 0], p[3, 1], 0,     1.,
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
    return GraphicsContext((1, 1))
