import sys
import unittest

try:
    from pyface.qt import QtGui
except ImportError:
    QT_NOT_AVAILABLE = True
else:
    QT_NOT_AVAILABLE = False
try:
    from pyface.qt import is_qt5
except ImportError:
    is_qt5 = False

from kiva.tests.drawing_tester import DrawingImageTester

is_linux = sys.platform.startswith("linux")


@unittest.skipIf(QT_NOT_AVAILABLE, "Cannot import qt")
class TestQPainterDrawing(DrawingImageTester, unittest.TestCase):
    def setUp(self):
        application = QtGui.QApplication.instance()
        if application is None:
            self.application = QtGui.QApplication([])
        else:
            self.application = application

        DrawingImageTester.setUp(self)

    def create_graphics_context(self, width, height):
        from kiva.qpainter import GraphicsContext

        return GraphicsContext((width, height))

    @unittest.skipIf(is_qt5 and is_linux, "Currently segfaulting")
    def test_text(self):
        super().test_text()

    @unittest.skipIf(is_qt5 and is_linux, "Currently segfaulting")
    def test_text_clip(self):
        super().test_text_clip()


if __name__ == "__main__":
    unittest.main()
