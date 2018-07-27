
from unittest import TestCase

try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock

from traits.api import Any
from traitsui.tests._tools import skip_if_not_wx

from enable.container import Container
from enable.base_tool import BaseTool
from enable.window import Window


class MouseEventTool(BaseTool):
    """ Tool that captures a single mouse wheel event """

    #: the captured mouse event
    event = Any

    def normal_mouse_wheel(self, event):
        self.event = event


@skip_if_not_wx
class MouseWheelTestCase(TestCase):

    def setUp(self):
        import wx

        # set up Enable components and tools
        self.container = Container(postion=[0, 0], bounds=[600, 600])
        self.tool = MouseEventTool(component=self.container)
        self.container.tools.append(self.tool)

        # set up wx components and tools
        self.parent = wx.Frame(None, size=(600, 600))
        self.window = Window(
            self.parent,
            size=(600, 600),
            component=self.container
        )

        # Hack: event processing code skips if window not actually shown by
        # testing for value of _size
        self.window._size = (600, 600)

    def test_vertical_mouse_wheel(self):
        import wx

        # create and mock a mouse wheel event
        wx_event = wx.MouseEvent(mouseType=wx.wxEVT_MOUSEWHEEL)
        wx_event.GetWheelRotation = MagicMock(return_value=200)
        wx_event.GetWheelAxis = MagicMock(return_value=wx.MOUSE_WHEEL_VERTICAL)
        wx_event.GetLinesPerAction = MagicMock(return_value=1)
        wx_event.GetWheelDelta = MagicMock(return_value=120)

        # dispatch event
        self.window._on_mouse_wheel(wx_event)

        # validate results
        self.assertEqual(self.tool.event.mouse_wheel_axis, 'vertical')
        self.assertAlmostEqual(self.tool.event.mouse_wheel, 5.0/3.0)
        self.assertEqual(self.tool.event.mouse_wheel_delta, (0, 200))

    def test_horizontal_mouse_wheel(self):
        import wx

        # create and mock a mouse wheel event
        wx_event = wx.MouseEvent(mouseType=wx.wxEVT_MOUSEWHEEL)
        wx_event.GetWheelRotation = MagicMock(return_value=200)
        wx_event.GetWheelAxis = MagicMock(
            return_value=wx.MOUSE_WHEEL_HORIZONTAL)
        wx_event.GetLinesPerAction = MagicMock(return_value=1)
        wx_event.GetWheelDelta = MagicMock(return_value=120)

        # dispatch event
        self.window._handle_mouse_event('mouse_wheel', wx_event)

        # validate results
        self.assertEqual(self.tool.event.mouse_wheel_axis, 'horizontal')
        self.assertAlmostEqual(self.tool.event.mouse_wheel, 5.0/3.0)
        self.assertEqual(self.tool.event.mouse_wheel_delta, (200, 0))
