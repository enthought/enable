# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style

def get_macport_qt(winId):
    """
    Returns the CGContextRef of an NSView instance.
    """
    from macport_qt import get_macport

    return get_macport(winId)

def get_macport_wx(dc):
    """
    Returns the m_macPort of a wxDC (or child class) instance.
    """
    from macport_wx import get_macport

    return get_macport(str(dc.this))
