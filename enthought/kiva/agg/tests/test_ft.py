from enthought.util.numerix import *
from enthought import freetype

import sys

from enthought.kiva import agg

ft = freetype.FreeType()
for i in range(100):
    ft.render("hello")

gc = agg.GraphicsContextArray((100,100))    
gc.set_line_dash(array((2.0,4.0,2.0,4.0,2.0,4.0)))
del gc
