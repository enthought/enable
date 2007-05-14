
import sys

if sys.platform == 'darwin':
    from mac_window import MacWindow as Window
else:
    from window import Window

from scrollbar import NativeScrollBar

del sys

