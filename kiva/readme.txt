[Importing Kiva]

To import kiva, you need to get the constants (which includes pen styles,
cap styles, lines styles, etc.), the font stuff, and a backend.  You can
import these all together with a wildcard:

    from kiva import *

If you prefer to get the constants by themselves, you can also do a:

    import kiva

and this will allow you to refer to them as kiva.JOIN_MITER, kiva.CAP_ROUND,
etc.

If you don't wish to do a wildcard import, you can also just grab the
following by themselves:

    from kiva import Font, CompiledPath, GraphicsContext,
                               Canvas, CanvasWindow

This will auto-detect what backend is appropriate based on the OS platform,
available libraries, and the order of backends listed in the KIVA_WISHLIST
environment variable.  If this environment variable is not defined, then the
default ordering of backends is in __init__.py.

If you want to choose what backend gets used, you can do the following:

    from kiva.backend_<backend> import GraphicsContext, Canvas, CanvasWindow

<backend> is one of:
    image   - in-memory GraphicsContext, uses Agg to raster, can save out to any PIL format
    wx    - uses Agg to raster to a platform-dependent wx window
    wx_gl - uses Agg to raster into a platform-dependent wx.glcanvas
    gl    - uses OpenGL module to draw into a GL window
    mac   - calls Quartz drawing routines into an OS X window/GraphicsContext
    ps    - PostScript file output
    svg   - SVG file output
    pdf   - PDF file output (uses ReportLab)


