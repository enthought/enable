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
""" Chaco's PostScript (PS/EPSF) backend

    :Copyright:   ActiveState
    :License:     BSD Style
    :Author:      David Ascher (davida@activestate.com)
    :Version:     $Revision: 1.2 $
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
from constants import FILL, STROKE, FILL_STROKE, EOF_FILL, EOF_FILL_STROKE

# This backend does not have compiled paths, yet.
CompiledPath = None

try:
    import logging
    import tempfile
    _logfile = os.path.join(tempfile.gettempdir(), "kivaps.log")
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
    constants.CAP_BUTT: 0,
    constants.CAP_ROUND: 1,
    constants.CAP_SQUARE: 2,
}

line_join_map = {
    constants.JOIN_MITER: 0,
    constants.JOIN_ROUND: 1,
    constants.JOIN_BEVEL: 2,
}

font_map = {'Arial': 'Helvetica'}

import _fontdata

font_map = {'Arial': 'Helvetica'}
try:
    # reportlab supports more fonts
    import reportlab.pdfbase.pdfmetrics as pdfmetrics
    import reportlab.pdfbase._fontdata as _fontdata
    _reportlab_loaded = 1
except ImportError:
    # we support the basic 14
    import pdfmetrics
    import _fontdata
    _reportlab_loaded = 0

font_face_map = {'Arial': 'Helvetica'}

_clip_counter = 0

fill_stroke_map = {FILL_STROKE: ('fill', 'stroke'),
                    EOF_FILL_STROKE: ('eofill', 'stroke'),
                    FILL: ('fill', None),
                    STROKE: ('stroke', None),
                    EOF_FILL: ('eofill', None)
                   }


class PSGC(basecore2d.GraphicsContextBase):

    def __init__(self, size):
        basecore2d.GraphicsContextBase.__init__(self)
        self.size = size
        self._height = size[1]
        self.contents = cStringIO.StringIO()
        self._clipmap = {}
        self.clip_id = None

    def save(self, filename):
        f = open(filename, 'w')
        ext = os.path.splitext(filename)[1]
        if ext in ('.eps', '.epsf'):
            f.write("%!PS-Adobe-3.0 EPSF-3.0\n")
            f.write('%%%%BoundingBox: 0 0 %d %d\n' % self.size)
            f.write(self.contents.getvalue())
        elif ext == '.ps':
            f.write("%!PS-Adobe-2.0\n")
            f.write(self.contents.getvalue())
        else:
            raise ValueError, "don't know how to write a %s file" % ext

    # Text handling code

    def set_font(self, font):
        self.face_name = font_face_map.get(font.face_name, font.face_name)
        self.font = pdfmetrics.Font(self.face_name, self.face_name, pdfmetrics.defaultEncoding)
        self.font_size = font.size
        self.contents.write("""/%s findfont %3.3f scalefont setfont\n""" % (self.face_name, self.font_size))

    def device_show_text(self, text):
        x,y = self.get_text_position()
        ttm = self.get_text_matrix()
        ctm = self.get_ctm()  # not device_ctm!!
        m = affine.concat(ctm,ttm)
        tx,ty,sx,sy,angle = affine.trs_factor(m)
        angle = '"%3.3f"' % (angle / pi * 180.)
        height = self.get_full_text_extent(text)[1]
        self.contents.write('%3.3f %3.3f moveto\n' % (x,y))
        r,g,b,a = self.state.line_color
        self.contents.write('%1.3f %1.3f %1.3f setrgbcolor\n' % (r,g,b) )
        self.contents.write('(%s) show\n' % text)
        #self.contents.write('<g transform="translate(%(x)f,%(y)f)">\n' % locals())
        #self.contents.write('<g transform="scale(1,-1)">\n')
        #self._emit('text', contents=text, rotate=angle, kw={'font-family':repr(self.font.fontName),
        #                                                'font-size': '"'+ str(self.font_size) + '"'})

    def get_full_text_extent(self, text):
        ascent,descent=_fontdata.ascent_descent[self.face_name]
        descent = (-descent) * self.font_size / 1000.0
        ascent = ascent * self.font_size / 1000.0
        height = ascent + descent
        width = pdfmetrics.stringWidth(text, self.face_name, self.font_size)
        return width, height, descent, height*1.2 # assume leading of 1.2*height

    # actual implementation =)

    def device_fill_points(self, points, mode):

        linecap = line_cap_map[self.state.line_cap]
        linejoin = line_join_map[self.state.line_join]
        dasharray = self._dasharray()
        if dasharray:
            self.contents.write('%s 0 setdash\n' % dasharray)
        self.contents.write('%3.3f setlinewidth\n' % self.state.line_width)
        self.contents.write('%d setlinecap\n' % linecap)
        self.contents.write('%d setlinejoin\n' % linejoin)
        self.contents.write('newpath\n')
        x,y = points[0]
        self.contents.write('    %3.3f %3.3f moveto\n' % (x,y))
        for (x,y) in points[1:]:
            self.contents.write('    %3.3f %3.3f lineto\n' % (x,y))


        first_pass, second_pass = fill_stroke_map[mode]

        if second_pass:
            if first_pass in ('fill', 'eofill'):
                r,g,b,a = self.state.fill_color
                self.contents.write('%1.3f %1.3f %1.3f setrgbcolor\n' % (r,g,b) )
            else:
                r,g,b,a = self.state.line_color
                self.contents.write('%1.3f %1.3f %1.3f setrgbcolor\n' % (r,g,b) )

            self.contents.write('gsave %s grestore %s\n' % (first_pass, second_pass))
        else:
            if second_pass in ('fill', 'eofill'):
                r,g,b,a = self.state.fill_color
                self.contents.write('%1.3f %1.3f %1.3f setrgbcolor\n' % (r,g,b) )
            else:
                r,g,b,a = self.state.line_color
                self.contents.write('%1.3f %1.3f %1.3f setrgbcolor\n' % (r,g,b) )
            self.contents.write(first_pass + '\n')

    def device_stroke_points(self, points, mode):
        # handled by device_fill_points
        pass

    def device_set_clipping_path(self, x, y, width, height):
        self.contents.write('%3.3f %3.3f %3.3f %3.3f rectclip\n' % (x,y,width*2.,height*2.))

    def device_destroy_clipping_path(self):
        self.contents.write('initclip\n')

    # utility routines

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
            return '[]'
        return '[ ' + dasharray + ' ]'

    # noops which seem to be needed

    def device_update_line_state(self):
        pass

    def device_update_fill_state(self):
        pass



class Canvas:
    def __init__(self, filename='', id = -1, size = (200,200)):
        self.filename = filename
        self._size = size
        self.gc = PSGC(size)

    def size(self):
        return self._size

    def save(self):
        self.gc.save(self.filename)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print "Usage: %s output_file (where output_file ends in .html or .svg" % sys.argv[0]
        raise SystemExit

