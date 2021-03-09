import unittest
from unittest import mock

from enable.abstract_window import AbstractWindow
from enable.component import Component


class TestAbstractWindow(unittest.TestCase):

    @mock.patch.object(AbstractWindow, "component_bounds_changed")
    def test_component_bounds_change(self, mock_method):
        class TestWindow(AbstractWindow):

            # needed to avoid NotImplementedError
            def _redraw(self):
                pass

            def _get_control_size(self):
                # this happens in the wild 
                return None
        
        thing = Component()

        TestWindow(
            parent=None,
            component=thing,
        )
        thing.bounds = [13.0, 13.0]
        
        self.assertTrue(mock_method.called)