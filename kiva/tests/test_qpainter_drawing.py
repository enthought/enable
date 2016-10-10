try:
    from pyface.qt import QtGui
except ImportError:
    QT_NOT_AVAILABLE = True
else:
    QT_NOT_AVAILABLE = False

from kiva.tests.drawing_tester import DrawingImageTester
from traits.testing.unittest_tools import unittest


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


if __name__ == "__main__":
    unittest.main()
