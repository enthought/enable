import unittest

from enable.tools.drag_tool import DragTool
from enable.events import KeyEvent
from enable.abstract_window import AbstractWindow


class DummyTool(DragTool):

    canceled = False

    def drag_cancel(self, event):
        self.canceled = True
        return True


class DummyWindow(AbstractWindow):

    mouse_owner = None

    def _get_control_size(self):
        return (0, 0)

    def _redraw(self):
        pass


class DragToolTestCase(unittest.TestCase):

    def setUp(self):
        self.tool = DummyTool()
        self.window = DummyWindow()

    def test_cancel_key_pressed(self):
        tool = self.tool
        tool._drag_state = 'dragging'  # force dragging state
        event = KeyEvent(character='Esc', event_type='key_pressed',
                         alt_down=False, control_down=False,
                         shift_down=False, window=self.window)
        tool.dispatch(event, 'key_pressed')
        self.assertTrue(tool.canceled)

    def test_cancel_keys_pressed(self):
        tool = self.tool
        tool._drag_state = 'dragging'  # force dragging state
        tool.cancel_keys = ['a', 'Left']
        event = KeyEvent(character='Esc', event_type='key_pressed',
                         alt_down=False, control_down=False,
                         shift_down=False, window=self.window)
        tool.dispatch(event, 'key_pressed')
        self.assertFalse(tool.canceled)
        tool._drag_state = 'dragging'  # force dragging state
        tool.canceled = False
        event = KeyEvent(character='a', event_type='key_pressed',
                         alt_down=False, control_down=False,
                         shift_down=False, window=self.window)
        tool.dispatch(event, 'key_pressed')
        self.assertTrue(tool.canceled)
        tool._drag_state = 'dragging'  # force dragging state
        tool.canceled = False
        event = KeyEvent(character='Left', event_type='key_pressed',
                         alt_down=False, control_down=False,
                         shift_down=False, window=self.window)
        tool.dispatch(event, 'key_pressed')
        self.assertTrue(tool.canceled)

    def test_any_key_pressed(self):
        tool = self.tool
        tool._drag_state = 'dragging'  # force dragging state
        event = KeyEvent(character='Left', event_type='key_pressed',
                         alt_down=False, control_down=False,
                         shift_down=False, window=self.window)
        tool.dispatch(event, 'key_pressed')
        self.assertFalse(tool.canceled)

if __name__ == '__main__':
    unittest.main()
