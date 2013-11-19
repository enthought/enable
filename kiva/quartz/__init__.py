# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style

try:
    from mac_context import get_mac_context
except ImportError:
    get_mac_context = None


def get_macport(dc):
    """
    Returns the Port or the CGContext of a wxDC (or child class) instance.
    """
    if 'GetCGContext' in dir(dc):
        ptr = dc.GetCGContext()
        return int(ptr)
    else:
        from macport import get_macport as _get_macport
        return _get_macport(str(dc.this))
