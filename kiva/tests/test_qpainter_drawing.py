import unittest

from pyface.qt import QtGui

from kiva.tests.drawing_tester import DrawingTester
from kiva.qpainter import GraphicsContext


class TestQPainterDrawing(DrawingTester, unittest.TestCase):

    def setUp(self):
        application = QtGui.QApplication.instance()
        if application is None:
            self.application = QtGui.QApplication([])
        else:
            self.application = application

        DrawingTester.setUp(self)
        self.gc.set_stroke_color((0.0, 0.0, 0.0))
        self.gc.set_fill_color((0.0, 0.0, 1.0))


    def create_graphics_context(self, width, height):
        return GraphicsContext((width, height))

    def save_to_file(self):
        self.gc.save(self.filename)


if __name__ == "__main__":
    unittest.main()
