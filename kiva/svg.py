# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Chaco's SVG backend

    :Copyright:   ActiveState
    :License:     BSD Style
    :Author:      David Ascher (davida@activestate.com)
    :Version:     $Revision: 1.5 $
"""

####
#
# Known limitations
#
# * BUG: Weird behavior with compound plots
# * Limitation: text widths are lousy if reportlab is not installed
# * Missing feature: rotated text

"""
Miscellaneous notes:

* the way to do links:
  <a xlink:href="http://www.w3.org">
    <ellipse cx="2.5" cy="1.5" rx="2" ry="1" fill="red" />
  </a>
"""
from base64 import b64encode
from io import BytesIO, StringIO
import os
import warnings

from numpy import arange, ndarray, ravel

# Local, relative Kiva imports
from . import affine
from . import basecore2d
from . import constants
from .constants import FILL, FILL_STROKE, EOF_FILL_STROKE, EOF_FILL, STROKE


def _strpoints(points):
    c = StringIO()
    for x, y in points:
        c.write("%3.2f,%3.2f " % (x, y))
    return c.getvalue()


def _mkstyle(kw):
    return "; ".join([str(k) + ":" + str(v) for k, v in kw.items()])


def default_filter(kw1):
    kw = {}
    for (k, v) in kw1.items():
        if isinstance(v, tuple):
            if v[0] != v[1]:
                kw[k] = v[0]
        else:
            kw[k] = v
    return kw


line_cap_map = {
    constants.CAP_ROUND: "round",
    constants.CAP_SQUARE: "square",
    constants.CAP_BUTT: "butt",
}

line_join_map = {
    constants.JOIN_ROUND: "round",
    constants.JOIN_BEVEL: "bevel",
    constants.JOIN_MITER: "miter",
}

xmltemplate = """<?xml version="1.0"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.0//EN"
"http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">
<svg xmlns="http://www.w3.org/2000/svg"
        xmlns:text="http://xmlns.graougraou.com/svg/text/"
        xmlns:a3="http://ns.adobe.com/AdobeSVGViewerExtensions/3.0/"
        xmlns:xlink="http://www.w3.org/1999/xlink"
        a3:scriptImplementation="Adobe"
        width="%(width)f"
        height="%(height)f"
        viewBox="0 0 %(width)f %(height)f"
        >
<g transform="translate(0,%(height)f)">
<g transform="scale(1,-1)">
%(contents)s
</g>
</g>
</svg>
"""

htmltemplate = """<html xmlns:svg="http://www.w3.org/2000/svg"
      xmlns:xlink="http://www.w3.org/1999/xlink">
<object id="AdobeSVG" CLASSID="clsid:78156a80-c6a1-4bbf-8e6a-3cd390eeb4e2">
</object>
<?import namespace="svg" implementation="#AdobeSVG"?>
<body>
<svg:svg width="100%%" height="100%%" viewBox="0 0 %(width)f %(height)f">
%(contents)s
</svg:svg>
</body>
</html>
"""

font_map = {"Arial": "Helvetica"}
try:
    # expensive way of computing string widths
    import reportlab.pdfbase.pdfmetrics as pdfmetrics
    import reportlab.pdfbase._fontdata as _fontdata

    _reportlab_loaded = 1
except ImportError:
    from . import pdfmetrics
    from . import _fontdata

    _reportlab_loaded = 0

font_face_map = {"Arial": "Helvetica", "": "Helvetica"}


# This backend has no compiled path object, yet.
class CompiledPath(object):
    pass


_clip_counter = 0


class GraphicsContext(basecore2d.GraphicsContextBase):
    def __init__(self, size, *args, **kwargs):
        super(GraphicsContext, self).__init__(self, size, *args, **kwargs)
        self.size = size
        self._height = size[1]
        self.contents = StringIO()
        self._clipmap = {}

    def render(self, format):
        assert format == "svg"
        height, width = self.size
        contents = (
            self.contents.getvalue()
            .replace("<svg:", "<")
            .replace("</svg:", "</")
        )
        return xmltemplate % locals()

    def clear(self):
        self.contents = StringIO()

    def width(self):
        return self.size[0]

    def height(self):
        return self.size[1]

    def save(self, filename):
        with open(filename, "w") as f:
            ext = os.path.splitext(filename)[1]
            if ext == ".svg":
                template = xmltemplate
                width, height = self.size
                contents = (
                    self.contents.getvalue()
                    .replace("<svg:", "<")
                    .replace("</svg:", "</")
                )
            elif ext == ".html":
                width, height = self.size[0] * 3, self.size[1] * 3
                contents = self.contents.getvalue()
                template = htmltemplate
            else:
                raise ValueError("don't know how to write a %s file" % ext)
            f.write(template % locals())

    # Text handling code

    def set_font(self, font):
        self.face_name = font_face_map.get(font.face_name, font.face_name)
        self.font = pdfmetrics.Font(
            self.face_name, self.face_name, pdfmetrics.defaultEncoding
        )
        self.font_size = font.size

    # actual implementation =)

    def device_show_text(self, text):
        ttm = self.get_text_matrix()
        ctm = self.get_ctm()  # not device_ctm!!
        m = affine.concat(ctm, ttm)
        # height = self.get_full_text_extent(text)[1]
        a, b, c, d, tx, ty = affine.affine_params(m)
        transform = (
            "matrix(%(a)f,%(b)f,%(c)f,%(d)f,%(tx)f,%(ty)f) scale(1,-1)"
            % locals()
        )
        self._emit(
            "text",
            contents=text,
            kw={
                "font-family": self.face_name,
                "font-size": str(self.font_size),
                "xml:space": "preserve",
                "transform": transform,
            },
        )

    def get_full_text_extent(self, text):
        ascent, descent = _fontdata.ascent_descent[self.face_name]
        descent = (-descent) * self.font_size / 1000.0
        ascent = ascent * self.font_size / 1000.0
        height = ascent + descent
        width = pdfmetrics.stringWidth(text, self.face_name, self.font_size)
        return (
            width,
            height,
            descent,
            height * 1.2,
        )  # assume leading of 1.2*height

    def device_draw_image(self, img, rect):
        """
        draw_image(img_gc, rect=(x,y,w,h))

        Draws another gc into this one.  If 'rect' is not provided, then
        the image gc is drawn into this one, rooted at (0,0) and at full
        pixel size.  If 'rect' is provided, then the image is resized
        into the (w,h) given and drawn into this GC at point (x,y).

        img_gc is either a Numeric array (WxHx3 or WxHx4) or a PIL Image.

        Requires the Python Imaging Library (PIL).
        """
        from PIL import Image

        # We turn img into a PIL object, since that is what ReportLab
        # requires.
        if isinstance(img, ndarray):
            # From numpy array
            pil_img = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            pil_img = img
        elif hasattr(img, "bmp_array"):
            # An offscreen kiva agg context
            if hasattr(img, "convert_pixel_format"):
                img = img.convert_pixel_format("rgba32", inplace=0)
            pil_img = Image.fromarray(img.bmp_array)
        else:
            warnings.warn(
                "Cannot render image of type %r into SVG context." % type(img)
            )
            return

        if rect is None:
            rect = (0, 0, pil_img.width, pil_img.height)

        left, top, width, height = rect
        if width != pil_img.width or height != pil_img.height:
            # This is not strictly required.
            pil_img = pil_img.resize((int(width), int(height)), Image.NEAREST)

        png_buffer = BytesIO()
        pil_img.save(png_buffer, "png")
        b64_img_data = b64encode(png_buffer.getvalue()).decode('utf8')
        png_buffer.close()

        # Draw the actual image.
        m = self.get_ctm()
        # Place the image on the page.
        # Using bottom instead of top here to account for the y-flip.
        m = affine.translate(m, left, height + top)
        transform = (
            "matrix(%f,%f,%f,%f,%f,%f) scale(1,-1)" % affine.affine_params(m)
        )
        # Flip y to reverse the flip at the start of the document.
        image_data = "data:image/png;base64," + b64_img_data
        self._emit(
            "image",
            transform=transform,
            width=str(width),
            height=str(height),
            preserveAspectRatio="none",
            kw={"xlink:href": image_data},
        )

    def device_fill_points(self, points, mode):
        points = self._fixpoints(points)
        if mode in (FILL, FILL_STROKE, EOF_FILL_STROKE):
            fill = self._color(self.state.fill_color)
        else:
            fill = "none"
        if mode in (STROKE, FILL_STROKE, EOF_FILL_STROKE):
            stroke = self._color(self.state.line_color)
        else:
            stroke = "none"
        if mode in (EOF_FILL_STROKE, EOF_FILL):
            rule = "evenodd"
        else:
            rule = "nonzero"
        linecap = line_cap_map[self.state.line_cap]
        linejoin = line_join_map[self.state.line_join]
        dasharray = self._dasharray()
        width = "%3.3f" % self.state.line_width
        clip_id = getattr(self.state, "_clip_id", None)
        if clip_id:
            clip = "url(#" + clip_id + ")"
        else:
            clip = None
        a, b, c, d, tx, ty = affine.affine_params(self.get_ctm())
        transform = "matrix(%(a)f,%(b)f,%(c)f,%(d)f,%(tx)f,%(ty)f)" % locals()
        if mode == STROKE:
            opacity = "%1.3f" % self.state.line_color[-1]
            self._emit(
                "polyline",
                transform=transform,
                points=_strpoints(points),
                kw=default_filter({"clip-path": (clip, None)}),
                style=_mkstyle(
                    default_filter(
                        {
                            "opacity": (opacity, "1.000"),
                            "stroke": stroke,
                            "fill": "none",
                            "stroke-width": (width, "1.000"),
                            "stroke-linejoin": (linejoin, "miter"),
                            "stroke-linecap": (linecap, "butt"),
                            "stroke-dasharray": (dasharray, "none"),
                        }
                    )
                ),
            )

        else:
            opacity = "%1.3f" % self.state.fill_color[-1]
            self._emit(
                "polygon",
                transform=transform,
                points=_strpoints(points),
                kw=default_filter({"clip-path": (clip, None)}),
                style=_mkstyle(
                    default_filter(
                        {
                            "opacity": (opacity, "1.000"),
                            "stroke-width": (width, "1.000"),
                            "fill": fill,
                            "fill-rule": rule,
                            "stroke": stroke,
                            "stroke-linejoin": (linejoin, "miter"),
                            "stroke-linecap": (linecap, "butt"),
                            "stroke-dasharray": (dasharray, "none"),
                        }
                    )
                ),
            )

    def device_stroke_points(self, points, mode):
        # handled by device_fill_points
        pass

    def _build(self, elname, contents=None, **kw):
        x = "<" + elname + " "
        for k, v in kw.items():
            if isinstance(v, float):
                v = "%3.3f" % v
            elif isinstance(v, int):
                v = "%d" % v
            else:
                v = "%s" % str(v)
            x += k + '="' + v + '" '
        if contents is None:
            x += "/>\n"
        else:
            x += ">"
            if elname != "text":
                x += "\n"
            x += contents
            x += "</" + elname + ">\n"
        return x

    def _debug_draw_clipping_path(self, x, y, width, height):
        a, b, c, d, tx, ty = affine.affine_params(self.get_ctm())
        transform = "matrix(%(a)f,%(b)f,%(c)f,%(d)f,%(tx)f,%(ty)f)" % locals()
        self._emit(
            "rect",
            x=x,
            y=y,
            width=width,
            height=height,
            transform=transform,
            style=_mkstyle(
                {"stroke-width": 5, "fill": "none", "stroke": "green"}
            ),
        )

    def device_set_clipping_path(self, x, y, width, height):
        # self._debug_draw_clipping_path(x, y, width, height)
        # return
        global _clip_counter
        self.state._clip_id = "clip_%d" % _clip_counter
        _clip_counter += 1
        x, y = self._fixpoints([[x, y]])[0]
        a, b, c, d, tx, ty = affine.affine_params(self.get_ctm())
        transform = "matrix(%(a)f,%(b)f,%(c)f,%(d)f,%(tx)f,%(ty)f)" % locals()
        rect = self._build("rect", x=x, y=y, width=width, height=height)
        clippath = self._build(
            "clipPath", contents=rect, id=self.state._clip_id
        )
        self._emit("g", transform=transform, contents=clippath)

    def device_destroy_clipping_path(self):
        self.state._clip_id = None

    # utility routines

    def _fixpoints(self, points):
        return points
        # convert lines from Kiva coordinate space to PIL coordinate space
        # XXX I suspect this is the location of the bug w.r.t. compound graphs
        # and "global" sizing.
        # XXX this should be made more efficient for NumPy arrays
        np = []
        for (x, y) in points:
            np.append((x, self._height - y))
        return np

    def _emit(self, name, contents=None, kw={}, **otherkw):
        self.contents.write("<svg:%(name)s " % locals())
        for k, v in kw.items():
            self.contents.write('%(k)s="%(v)s" ' % locals())
        for k, v in otherkw.items():
            self.contents.write('%(k)s="%(v)s" ' % locals())
        if contents is None:
            self.contents.write("/>\n")
        else:
            self.contents.write(">")
            if name != "text":
                self.contents.write("\n")
            self.contents.write(contents)
            self.contents.write("</svg:" + name + ">\n")

    def _color(self, color):
        r, g, b, a = color
        return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

    def _dasharray(self):
        dasharray = ""
        for x in self.state.line_dash:
            if type(x) == type(arange(3)):  # why is this so hard?
                x = ravel(x)[0]
            dasharray += " " + "%3.2f" % x
        if not dasharray or dasharray == " 0.00 0.00":
            dasharray = "none"
        return dasharray

    # noops which seem to be needed

    def device_update_line_state(self):
        pass

    def device_update_fill_state(self):
        pass


def font_metrics_provider():
    return GraphicsContext((1, 1))


SVGGC = GraphicsContext  # for b/w compatibility
