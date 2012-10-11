import unittest

from traits.api import Int

from enable.tools.drag_tool import DragTool
from enable.events import KeyEvent, MouseEvent
from enable.abstract_window import AbstractWindow


class DummyTool(DragTool):

    canceled = Int

    def drag_cancel(self, event):
        self.canceled += 1
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
        self.assertEqual(tool.canceled, 1)

    def test_multiple_cancel_keys(self):
        tool = self.tool

        tool._drag_state = 'dragging'  # force dragging state
        tool.cancel_keys = ['a', 'Left']
        self._press_key('Esc')
        self.assertEqual(tool.canceled, 0)

        tool._drag_state = 'dragging'  # force dragging state
        self._press_key('a')
        self.assertTrue(tool.canceled)

        tool._drag_state = 'dragging'  # force dragging state
        self._press_key('Left')
        self.assertEqual(tool.canceled, 2)

    def test_other_key_pressed(self):
        tool = self.tool
        tool._drag_state = 'dragging'  # force dragging state
        self._press_key('Left')
        self.assertEqual(tool.canceled, 0)

    def test_mouse_leave_drag_state(self):

        # check when end_drag_on_leave is true
        tool = self.tool
        tool.end_drag_on_leave = True
        tool._drag_state = 'dragging'  # force dragging state
        event = MouseEvent(x=0, y=0, window=self.window)
        self.tool.dispatch(event, 'mouse_leave')
        self.assertEqual(tool.canceled, 1)
        self.assertEqual(tool._drag_state, 'nondrag')


        # check when end_drag_on_leave is false
        tool.end_drag_on_leave = False
        tool._drag_state = 'dragging'  # force dragging state
        event = MouseEvent(x=0, y=0, window=self.window)
        self.tool.dispatch(event, 'mouse_leave')
        self.assertEqual(tool.canceled, 1)
        self.assertEqual(tool._drag_state, 'dragging')

if __name__ == '__main__':
    unittest.main()
