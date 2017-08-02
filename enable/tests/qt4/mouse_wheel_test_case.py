
from unittest import TestCase

from traits.api import Any
from traitsui.tests._tools import skip_if_not_qt4

from enable.container import Container
from enable.base_tool import BaseTool
from enable.window import Window


class MouseEventTool(BaseTool):
    """ Tool that captures a single mouse wheel event """

    #: the captured mouse event
    event = Any

    def normal_mouse_wheel(self, event):
        self.event = event


@skip_if_not_qt4
class MouseWheelTestCase(TestCase):

    def setUp(self):

        # set up Enable components and tools
        self.container = Container(postion=[0, 0], bounds=[600, 600])
        self.tool = MouseEventTool(component=self.container)
        self.container.tools.append(self.tool)

        # set up qt components
        self.window = Window(
            None,
            size=(600, 600),
            component=self.container
        )

        # Hack: event processing code skips if window not actually shown by
        # testing for value of _size
        self.window._size = (600, 600)

    def test_vertical_mouse_wheel(self):
        from pyface.qt import QtCore, QtGui

        # create and mock a mouse wheel event
        qt_event = QtGui.QWheelEvent(
            QtCore.QPoint(0, 0), 200, QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
            QtCore.Qt.Vertical
        )

        # dispatch event
        self.window._on_mouse_wheel(qt_event)

        # validate results
        self.assertEqual(self.tool.event.mouse_wheel_axis, 'vertical')
        self.assertAlmostEqual(self.tool.event.mouse_wheel, 5.0/3.0)

    def test_horizontal_mouse_wheel(self):
        from pyface.qt import QtCore, QtGui

        # create and mock a mouse wheel event
        qt_event = QtGui.QWheelEvent(
            QtCore.QPoint(0, 0), 200, QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
            QtCore.Qt.Horizontal
        )

        # dispatch event
        self.window._on_mouse_wheel(qt_event)

        # validate results
        self.assertEqual(self.tool.event.mouse_wheel_axis, 'horizontal')
        self.assertAlmostEqual(self.tool.event.mouse_wheel, 5.0/3.0)
