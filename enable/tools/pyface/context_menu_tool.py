# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" This is a minimum example for adding a context menu to plot.

"""

from pyface.action.api import MenuManager, ActionController

from traits.api import Instance, Any
from enable.base_tool import BaseTool


class EnableActionController(ActionController):
    """ An action controller that keeps a reference to the enable event that
    triggered the action.

    """

    #: the enable event which triggered the popup menu
    enable_event = Any

    def perform(self, action, event):
        """ Control an action invocation

        We make the original enable event available to the action by adding it
        to the pyface event.

        """
        event.enable_event = self.enable_event
        return action.perform(event)


class ContextMenuTool(BaseTool):
    """ Pops up a context menu when the component receives a right click
    """

    #: the pyface action MenuManager instance
    menu_manager = Instance(MenuManager)

    #: an optional ActionController
    controller = Instance(ActionController)

    def normal_right_down(self, event):
        """ Handles the right mouse button being pressed.
        """
        if self.menu_manager is not None:
            if self.is_showable(event.x, event.y):
                self.show_menu(event)
                event.handled = True

    def is_showable(self, x, y):
        """ Returns whether the (x, y) position is OK for showing the menu

        By default checks that the point is in the component.  Subclasses can
        override to provide more refined hit-testing.

        """
        return self.component.is_in(x, y)

    def show_menu(self, event):
        """ Create the toolkit menu and show it

        This method also makes the enable event available to the controller.

        """
        controller = self.controller
        if controller is None:
            controller = self.menu_manager.controller
            if controller is None:
                controller = EnableActionController(enable_event=event)
        else:
            controller.enable_event = event
        menu = self.menu_manager.create_menu(event.window.control, controller)
        menu.show()
