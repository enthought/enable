# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import unittest
from unittest import mock

from enable.abstract_window import AbstractWindow
from enable.component import Component


class TestAbstractWindow(unittest.TestCase):

    @mock.patch.object(AbstractWindow, "component_bounds_updated")
    def test_component_bounds_updated(self, mock_method):
        """ Make sure trait listener for changing component bounds gets set up.
        """

        class TestWindow(AbstractWindow):
            # needed to avoid a NotImplementedError, not under test
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
