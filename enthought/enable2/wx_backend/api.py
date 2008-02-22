
import sys

from enthought.kiva import backend

if sys.platform == 'darwin' and backend() != "gl":
    from mac_window import MacWindow as Window
else:
    from window import Window

from scrollbar import NativeScrollBar

del sys

