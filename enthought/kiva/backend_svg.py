#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#------------------------------------------------------------------------------
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

# Major library imports
import os
import sys
import cStringIO
from numpy import arange, ravel, pi

# Local, relative Kiva imports
import affine
import basecore2d
import constants
from constants import FILL, FILL_STROKE, EOF_FILL_STROKE, EOF_FILL, STROKE

try:
    import logging
    import tempfile
    _logfile = os.path.join(tempfile.gettempdir(), "kivasvg.log")
    hdlr = logging.FileHandler(_logfile)
    BASIC_FORMAT = "%(levelname)s: %(name)s: %(message)s"
    fmt = logging.Formatter(BASIC_FORMAT)
    hdlr.setFormatter(fmt)
    logging.root.addHandler(hdlr)
    log = logging.getLogger('')
    log.setLevel(logging.INFO)
except ImportError:
    class FakeLogger:
        def debug(self, message):
            print >> sys.stderr, "DEBUG:", message
        def info(self, message):
            print >> sys.stderr, "INFO:", message
        def warn(self, message):
            print >> sys.stderr, "WARN:", message
        def error(self, message):
            print >> sys.stderr, "ERROR:", message
        def critical(self, message):
            print >> sys.stderr, "CRITICAL:", message
    log = FakeLogger()

def _strpoints(points):
    c = cStringIO.StringIO()
    for x,y in points:
        c.write('%3.2f,%3.2f ' % (x,y))
    return c.getvalue()

def _mkstyle(kw):
    return '"' + '; '.join([str(k) + ':' + str(v) for k,v in kw.items()]) +'"'


def default_filter(kw1):
    kw = {}
    for (k,v) in kw1.items():
        if type(v) == type(()):
            if v[0] != v[1]:
                kw[k] = v[0]
        else:
            kw[k] = v
    return kw

line_cap_map = {
    constants.CAP_ROUND: 'round',
    constants.CAP_SQUARE: 'square',
    constants.CAP_BUTT: 'butt'
    }

line_join_map = {
    constants.JOIN_ROUND: 'round',
    constants.JOIN_BEVEL: 'bevel',
    constants.JOIN_MITER: 'miter'
    }

font_map = {'Arial': 'Helvetica',
            }
import _fontdata

xmltemplate = """<?xml version="1.0"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.0//EN"
"http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">
<svg xmlns="http://www.w3.org/2000/svg"
        xmlns:text="http://xmlns.graougraou.com/svg/text/"
        xmlns:a3="http://ns.adobe.com/AdobeSVGViewerExtensions/3.0/"
        a3:scriptImplementation="Adobe"
        width="100%%"
        height="100%%"
        viewBox="0 0 %(width)f %(height)f"
        >
<g transform="translate(0,%(height)f)">
<g transform="scale(1,-1)">
%(contents)s
</g>
</g>
</svg>
"""

htmltemplate = """<html xmlns:svg="http://www.w3.org/2000/svg">
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

font_map = {'Arial': 'Helvetica',
            }
try:
    # expensive way of computing string widths
    import reportlab.pdfbase.pdfmetrics as pdfmetrics
    import reportlab.pdfbase._fontdata as _fontdata
    _reportlab_loaded = 1
except ImportError:
    import pdfmetrics
    import _fontdata
    _reportlab_loaded = 0

font_face_map = {'Arial': 'Helvetica'}

# This backend has no compiled path object, yet.
class CompiledPath(object):
    pass

_clip_counter = 0
class GraphicsContext(basecore2d.GraphicsContextBase):
    
    def __init__(self, size):
        basecore2d.GraphicsContextBase.__init__(self)
        self.size = size
        self._height = size[1]
        self.contents = cStringIO.StringIO()
        self._clipmap = {}
        self.clip_id = None

    def render(self, format):
        assert format == 'svg'
        height, width = self.size
        contents = self.contents.getvalue().replace("<svg:", "<").replace("</svg:", "</")
        return xmltemplate % locals()

    def clear(self, size):
        # TODO: clear the contents
        pass
    
    def width(self):
        return self.size[0]
    
    def height(self):
        return self.size[1]
    
    def save(self, filename):
        f = open(filename, 'w')
        ext = os.path.splitext(filename)[1]
        if ext == '.svg':
            template = xmltemplate
            height, width = self.size
            contents = self.contents.getvalue().replace("<svg:", "<").replace("</svg:", "</")
        elif ext == '.html':
            height, width = self.size[0]*3, self.size[1]*3
            contents = self.contents.getvalue()
            template = htmltemplate
        else:
            raise ValueError, "don't know how to write a %s file" % ext
        f.write(template % locals())
        

    # Text handling code

    def set_font(self, font):
        if font.face_name == '':
            font.face_name = 'Arial'
        self.face_name = font_face_map.get(font.face_name, font.face_name)
        self.font = pdfmetrics.Font(self.face_name, self.face_name, pdfmetrics.defaultEncoding)
        self.font_size = font.size

    def device_show_text(self, text):
        x,y = self.get_text_position()
        x,y = self._fixpoints([[x,y]])[0]
        ttm = self.get_text_matrix()
        ctm = self.get_ctm()  # not device_ctm!!
        m = affine.concat(ctm,ttm)
        tx,ty,sx,sy,angle = affine.trs_factor(m)
        angle = '%3.3f' % (-angle / pi * 180.)
        height = self.get_full_text_extent(text)[1]
        self.contents.write('<g transform="translate(%(x)f,%(y)f)">\n' % locals())
        self.contents.write('<g transform="scale(1,-1)">\n')
        self._emit('text', contents=text, transform='"rotate('+angle+')"', kw={'font-family':repr(self.font.fontName),
                                                        'font-size': '"'+ str(self.font_size) + '"'})
        self.contents.write('</g>\n')
        self.contents.write('</g>\n')
    def get_full_text_extent(self, text):
        ascent,descent=_fontdata.ascent_descent[self.face_name]
        descent = (-descent) * self.font_size / 1000.0
        ascent = ascent * self.font_size / 1000.0
        height = ascent + descent
        width = pdfmetrics.stringWidth(text, self.face_name, self.font_size)
        return width, height, descent, height*1.2 # assume leading of 1.2*height

    # actual implementation =)

    def device_fill_points(self, points, mode):
        points = self._fixpoints(points)
        if mode in (FILL, FILL_STROKE, EOF_FILL_STROKE):
            fill = self._color(self.state.fill_color)
        else:
            fill = 'none'
        if mode in (STROKE, FILL_STROKE, EOF_FILL_STROKE):
            stroke = self._color(self.state.line_color)
        else:
            stroke = 'none'
        if mode in (EOF_FILL_STROKE, EOF_FILL):
            rule = 'evenodd'
        else:
            rule = 'nonzero'
        linecap = line_cap_map[self.state.line_cap]
        linejoin = line_join_map[self.state.line_join]
        dasharray = self._dasharray()
        width = '%3.3f' % self.state.line_width
        if self.clip_id:
            clip = '"url(#' + self.clip_id +')"'
        else:
            clip = None
        if mode == STROKE:
            opacity = '%1.3f' % self.state.line_color[-1]
            self._emit('polyline',
                        points='"'+_strpoints(points)+'"',
                        kw=default_filter({'clip-path': (clip, None)}),
                        style=_mkstyle(default_filter({'opacity': (opacity, "1.000"),
                                        'stroke': stroke,
                                        'fill': 'none',
                                        'stroke-width': (width, "1.000"),
                                        'stroke-linejoin': (linejoin, 'miter'),
                                        'stroke-linecap': (linecap, 'butt'),
                                        'stroke-dasharray': (dasharray, 'none')})))

        else:
            opacity = '%1.3f' % self.state.fill_color[-1]
            self._emit('polygon',
                        points='"'+_strpoints(points)+'"',
                        kw=default_filter({'clip-path': (clip, None)}),
                        style=_mkstyle(default_filter({'opacity': (opacity, "1.000"),
                                        'stroke-width': (width, "1.000"),
                                        'fill': fill,
                                        'fill-rule': rule,
                                        'stroke': stroke,
                                        'stroke-linejoin': (linejoin, 'miter'),
                                        'stroke-linecap': (linecap, 'butt'),
                                        'stroke-dasharray': (dasharray, 'none')})))

    def device_stroke_points(self, points, mode):
        # handled by device_fill_points
        pass

    def _build(self, elname, **kw):
        x = '<' + elname + ' '
        for k,v in kw.items():
            if type(v) == type(0.0):
                v = '"%3.3f"' % v
            elif type(v) == type(0):
                v = '"%d"' % v
            else:
                v = '"%s"' % str(v)
            x += k + '=' + v + ' '
        x += '/>\n'
        return x

    def device_set_clipping_path(self, x, y, width, height):
        ##x,y,width,height = map(lambda x: '"' + str(x) + '"', [x,y,width,height])
        ##self._emit('rect', x=x, y=y, width=width, height=height,
        ##                style=_mkstyle({'stroke-width': 5,
        ##                                'fill':'none',
        ##                                'stroke': 'green'}))
        ##
        ##return
        global _clip_counter
        self.clip_id = 'clip_%d' % _clip_counter
        _clip_counter += 1
        x,y = self._fixpoints([[x,y]])[0]
        rect = self._build('rect', x=x, y=y, width=width, height=height)
        self._emit('clipPath', contents=rect, id='"'+self.clip_id + '"')

    def device_destroy_clipping_path(self):
        self.clip_id = None

    # utility routines

    def _fixpoints(self, points):
        return points
        # convert lines from Kiva coordinate space to PIL coordinate space
        # XXX I suspect this is the location of the bug w.r.t. compound graphs and
        # "global" sizing.
        # XXX this should be made more efficient for NumPy arrays
        np = []
        for (x,y) in points:
            np.append((x,self._height-y))
        return np

    def _emit(self, name, contents=None, kw={}, **otherkw):
        self.contents.write('<svg:%(name)s ' % locals())
        for k, v in kw.items():
            self.contents.write("%(k)s=%(v)s " % locals())
        for k, v in otherkw.items():
            self.contents.write("%(k)s=%(v)s " % locals())
        if contents is None:
            self.contents.write('/>\n')
        else:
            self.contents.write('>\n')
            self.contents.write(contents)
            self.contents.write('</svg:'+name+'>\n')

    def _color(self, color):
        r,g,b,a = color
        return '#%02x%02x%02x' % (r*255,g*255,b*255)

    def _dasharray(self):
        dasharray = ''
        for x in self.state.line_dash:
            if type(x) == type(arange(3)):  # why is this so hard?
                x = ravel(x)[0]
            dasharray += ' ' + '%3.2f' % x
        if not dasharray or dasharray == " 0.00 0.00":
            dasharray = 'none'
        return dasharray

    # noops which seem to be needed

    def device_update_line_state(self):
        pass

    def device_update_fill_state(self):
        pass


def font_metrics_provider():
    return GraphicsContext((1,1))

class Canvas:
    def __init__(self, filename='', id = -1, size = (200,200)):
        self.filename = filename
        self._size = size
        self.gc = SVGGC(size)

    def size(self):
        return self._size

    def save(self):
        self.gc.save(self.filename)

SVGGC = GraphicsContext # for b/w compatibility

class CanvasWindow: # required by core2d.py import, not sure why
    pass


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print "Usage: %s output_file (where output_file ends in .html or .svg" % sys.argv[0]
        raise SystemExit

