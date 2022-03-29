# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from unittest import TestCase

from traits.api import Any
from enable.tests._testing import skip_if_not_qt

from enable.container import Container
from enable.base_tool import BaseTool
from enable.window import Window


class MouseEventTool(BaseTool):
    """ Tool that captures a single mouse wheel event """

    #: the captured mouse event
    event = Any

    def normal_mouse_wheel(self, event):
        self.event = event


@skip_if_not_qt
class MouseWheelTestCase(TestCase):
    def setUp(self):

        # set up Enable components and tools
        self.container = Container(postion=[0, 0], bounds=[600, 600])
        self.tool = MouseEventTool(component=self.container)
        self.container.tools.append(self.tool)

        # set up qt components
        self.window = Window(None, size=(600, 600), component=self.container)

        # Hack: event processing code skips if window not actually shown by
        # testing for value of _size
        self.window._size = (600, 600)

    def test_vertical_mouse_wheel(self):
        from pyface.qt import QtCore, QtGui, is_qt4, is_qt5


        # create and mock a mouse wheel event
        if is_qt4:
            qt_event = QtGui.QWheelEvent(
                QtCore.QPoint(0, 0),
                200,
                QtCore.Qt.NoButton,
                QtCore.Qt.NoModifier,
                QtCore.Qt.Vertical,
            )
        elif is_qt5:
            qt_event = QtGui.QWheelEvent(
                pos=QtCore.QPointF(0, 0),
                globalPos=self.window.control.mapToGlobal(QtCore.QPoint(0, 0)),
                pixelDelta=QtCore.QPoint(0, 200),
                angleDelta=QtCore.QPoint(0, 200),
                qt4Delta=200,
                qt4Orietation=QtCore.Qt.Vertical,
                buttons=QtCore.Qt.NoButton,
                modifiers=QtCore.Qt.NoModifier,
                phase=QtCore.Qt.ScrollUpdate,
            )
        else:
            qt_event = QtGui.QWheelEvent(
                pos=QtCore.QPointF(0, 0),
                globalPos=self.window.control.mapToGlobal(QtCore.QPointF(0, 0)),
                pixelDelta=QtCore.QPoint(0, 200),
                angleDelta=QtCore.QPoint(0, 200),
                buttons=QtCore.Qt.NoButton,
                modifiers=QtCore.Qt.NoModifier,
                phase=QtCore.Qt.ScrollUpdate,
                inverted=False,
            )

        # dispatch event
        self.window._on_mouse_wheel(qt_event)

        # validate results
        self.assertEqual(self.tool.event.mouse_wheel_axis, "vertical")
        self.assertAlmostEqual(self.tool.event.mouse_wheel, 5.0 / 3.0)
        self.assertEqual(self.tool.event.mouse_wheel_delta, (0, 200))

    def test_horizontal_mouse_wheel(self):
        from pyface.qt import QtCore, QtGui, is_qt4, is_qt5

        # create and mock a mouse wheel event
        if is_qt4:
            qt_event = QtGui.QWheelEvent(
                QtCore.QPoint(0, 0),
                200,
                QtCore.Qt.NoButton,
                QtCore.Qt.NoModifier,
                QtCore.Qt.Horizontal,
            )
        elif is_qt5:
            qt_event = QtGui.QWheelEvent(
                pos=QtCore.QPointF(0, 0),
                globalPos=self.window.control.mapToGlobal(QtCore.QPoint(0, 0)),
                pixelDelta=QtCore.QPoint(200, 0),
                angleDelta=QtCore.QPoint(200, 0),
                qt4Delta=200,
                qt4Orietation=QtCore.Qt.Horizontal,
                buttons=QtCore.Qt.NoButton,
                modifiers=QtCore.Qt.NoModifier,
                phase=QtCore.Qt.ScrollUpdate,
            )
        else:
            qt_event = QtGui.QWheelEvent(
                pos=QtCore.QPointF(0, 0),
                globalPos=self.window.control.mapToGlobal(QtCore.QPointF(0, 0)),
                pixelDelta=QtCore.QPoint(200, 0),
                angleDelta=QtCore.QPoint(200, 0),
                buttons=QtCore.Qt.NoButton,
                modifiers=QtCore.Qt.NoModifier,
                phase=QtCore.Qt.ScrollUpdate,
                inverted=False,
            )

        # dispatch event
        self.window._on_mouse_wheel(qt_event)

        # validate results
        self.assertEqual(self.tool.event.mouse_wheel_axis, "horizontal")
        self.assertAlmostEqual(self.tool.event.mouse_wheel, 5.0 / 3.0)
        self.assertEqual(self.tool.event.mouse_wheel_delta, (200, 0))

    def test_vertical_mouse_wheel_without_pixel_delta(self):
        from pyface.qt import QtCore, QtGui, is_qt4, is_qt5

        # create and mock a mouse wheel event
        if is_qt4:
            qt_event = QtGui.QWheelEvent(
                QtCore.QPoint(0, 0),
                200,
                QtCore.Qt.NoButton,
                QtCore.Qt.NoModifier,
                QtCore.Qt.Vertical,
            )
        elif is_qt5:
            qt_event = QtGui.QWheelEvent(
                pos=QtCore.QPointF(0, 0),
                globalPos=self.window.control.mapToGlobal(QtCore.QPoint(0, 0)),
                pixelDelta=QtCore.QPoint(0, 0),
                angleDelta=QtCore.QPoint(0, 200),
                qt4Delta=200,
                qt4Orientation=QtCore.Qt.Horizontal,
                buttons=QtCore.Qt.NoButton,
                modifiers=QtCore.Qt.NoModifier,
                phase=QtCore.Qt.ScrollUpdate,
            )
        else:
            qt_event = QtGui.QWheelEvent(
                pos=QtCore.QPointF(0, 0),
                globalPos=self.window.control.mapToGlobal(QtCore.QPointF(0, 0)),
                pixelDelta=QtCore.QPoint(0, 0),
                angleDelta=QtCore.QPoint(0, 200),
                buttos=QtCore.Qt.NoButton,
                modifiers=QtCore.Qt.NoModifier,
                phase=QtCore.Qt.ScrollUpdate,
                inverted=False,
            )

        # dispatch event
        self.window._on_mouse_wheel(qt_event)

        # validate results
        self.assertEqual(self.tool.event.mouse_wheel_axis, "vertical")
        self.assertEqual(self.tool.event.mouse_wheel, 5.0 / 3.0)
        self.assertEqual(self.tool.event.mouse_wheel_delta, (0, 200))
