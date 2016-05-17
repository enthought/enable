
from contextlib import contextmanager

from traits.testing.unittest_tools import unittest
from traitsui.tests._tools import skip_if_not_qt4
from pyface.gui import GUI

from enable.container import Container
from enable.window import Window

try:
    from enable.qt4.scrollbar import NativeScrollBar
except Exception:
    NativeScrollBar = None


@skip_if_not_qt4
class ScrollBarTest(unittest.TestCase):

    def setUp(self):
        from pyface.qt.QtGui import QApplication
        from pyface.ui.qt4.util.event_loop_helper import EventLoopHelper

        qt_app = QApplication.instance()
        if qt_app is None:
            qt_app = QApplication([])
        self.qt_app = qt_app

        if NativeScrollBar is None:
            raise unittest.SkipTest("Qt4 NativeScrollbar not available.")
        self.gui = GUI()
        self.event_loop_helper = EventLoopHelper(gui=self.gui, qt_app=qt_app)
        self.container = Container(position=[0, 0], bounds=[600, 600])
        self.window = Window(None, size=(600, 600), component=self.container)

    @contextmanager
    def setup_window(self, window):
        window.control.show()
        window._size = window._get_control_size()
        self.gui.process_events()
        try:
            yield
        finally:
            self.gui.process_events()
            with self.event_loop_helper.delete_widget(window.control, timeout=1.0):
                window.control.deleteLater()

    @contextmanager
    def setup_scrollbar(self, scrollbar, window):
        scrollbar._draw_mainlayer(window._gc)
        try:
            yield
        finally:
            scrollbar.destroy()

    def test_scroll_position_horizontal(self):
        bounds = [600.0, 30.0]
        position = [0.0, 0.0]
        range = [600, 0.0, 375.0, 20.454545454545453]
        scrollbar = NativeScrollBar(
            orientation='horizontal',
            bounds=bounds,
            position=position,
            range=range,
        )
        self.container.add(scrollbar)
        with self.setup_window(self.window):
            with self.setup_scrollbar(scrollbar, self.window):
                self.assertEqual(scrollbar._control.value(), 0)
                self.assertEqual(scrollbar.scroll_position, 0)

                # move the scrollbar
                scrollbar._control.setValue(100)
                self.assertEqual(scrollbar.scroll_position, 100)

                # set the scroll & redraw
                scrollbar.scroll_position = 200
                scrollbar._draw_mainlayer(self, self.window._gc)
                self.assertEqual(scrollbar._control.value(), 200)

    def test_scroll_position_vertical(self):
        bounds = [30.0, 600.0]
        position = [0.0, 0.0]
        range = [600, 0.0, 375.0, 20.454545454545453]
        scrollbar = NativeScrollBar(
            orientation='vertical',
            bounds=bounds,
            position=position,
            range=range,
        )
        self.container.add(scrollbar)
        with self.setup_window(self.window):
            with self.setup_scrollbar(scrollbar, self.window):
                self.assertEqual(scrollbar._control.value(), 600-375)
                self.assertEqual(scrollbar.scroll_position, 0)

                # move the scrollbar
                scrollbar._control.setValue(100)
                self.assertEqual(scrollbar.scroll_position, 600-375-100)

                # set the scroll & redraw
                scrollbar.scroll_position = 200
                scrollbar._draw_mainlayer(self, self.window._gc)
                self.assertEqual(scrollbar._control.value(), 600-375-200)

    def test_minumum_horizontal(self):
        bounds = [600.0, 30.0]
        position = [0.0, 0.0]
        range = [700, 100.0, 375.0, 20.454545454545453]
        scrollbar = NativeScrollBar(
            orientation='horizontal',
            bounds=bounds,
            position=position,
            range=range,
        )
        self.container.add(scrollbar)
        with self.setup_window(self.window):
            with self.setup_scrollbar(scrollbar, self.window):
                self.assertEqual(scrollbar._control.value(), 100)
                self.assertEqual(scrollbar.scroll_position, 100)

                # move the scrollbar
                scrollbar._control.setValue(200)
                self.assertEqual(scrollbar.scroll_position, 200)

                # set the scroll & redraw
                scrollbar.scroll_position = 300
                scrollbar._draw_mainlayer(self, self.window._gc)
                self.assertEqual(scrollbar._control.value(), 300)

    def test_minimum_vertical(self):
        bounds = [30.0, 600.0]
        position = [0.0, 0.0]
        range = [700, 100.0, 375.0, 20.454545454545453]
        scrollbar = NativeScrollBar(
            orientation='vertical',
            bounds=bounds,
            position=position,
            range=range,
        )
        self.container.add(scrollbar)
        with self.setup_window(self.window):
            with self.setup_scrollbar(scrollbar, self.window):
                # control should be at top
                self.assertEqual(scrollbar._control.value(), 700-375)
                self.assertEqual(scrollbar.scroll_position, 100)

                # move the scrollbar to the bottom
                scrollbar._control.setValue(100)
                self.assertEqual(scrollbar.scroll_position, 700-375)

                # set the scroll & redraw
                scrollbar.scroll_position = 200
                scrollbar._draw_mainlayer(self, self.window._gc)
                self.assertEqual(scrollbar._control.value(), 700-375-(200-100))
