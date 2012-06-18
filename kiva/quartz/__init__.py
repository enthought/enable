# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style

from mac_context import get_mac_context
try:
    from macport import get_macport as _get_macport
except ImportError:
    # macport is not available on 64-bit.
    _get_macport = None


def get_macport(dc):
    """ Returns the m_macPort of a wxDC (or child class) instance.
        NOTE: This is only available on 32-bit at this time.
    """
    return _get_macport(str(dc.this))
