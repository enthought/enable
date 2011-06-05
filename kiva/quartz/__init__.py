# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style

from macport import get_macport as _get_macport

def get_macport(dc):
    """
    Returns the m_macPort of a wxDC (or child class) instance.
    """
    return _get_macport(str(dc.this))
