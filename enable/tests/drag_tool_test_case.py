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

    def _press_key(self, character):
        """ Create a key_pressed event and dispatch it. """

        event = KeyEvent(character=character, event_type='key_pressed',
                         alt_down=False, control_down=False,
                         shift_down=False, window=self.window)

        self.tool.dispatch(event, 'key_pressed')

    def test_default_cancel_key(self):
        tool = self.tool
        tool._drag_state = 'dragging'  # force dragging state

        self._press_key('Esc')

        self.assertTrue(tool.canceled)

    def test_multiple_cancel_keys(self):
        tool = self.tool

        tool._drag_state = 'dragging'  # force dragging state
        tool.cancel_keys = ['a', 'Left']
        self._press_key('Esc')
        self.assertFalse(tool.canceled)

        tool._drag_state = 'dragging'  # force dragging state
        tool.canceled = False
        self._press_key('a')
        self.assertTrue(tool.canceled)

        tool._drag_state = 'dragging'  # force dragging state
        tool.canceled = False
        self._press_key('Left')
        self.assertTrue(tool.canceled)

    def test_other_key_pressed(self):
        tool = self.tool
        tool._drag_state = 'dragging'  # force dragging state
        self._press_key('Left')
        self.assertFalse(tool.canceled)

if __name__ == '__main__':
    unittest.main()
