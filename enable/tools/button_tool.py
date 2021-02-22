# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""
ButtonTool
==========

A simple tool that responds to mouse clicks.
"""

from traits.api import Bool, Event

from enable.base_tool import BaseTool


class ButtonTool(BaseTool):
    """ A button tool

    This tool allows any component to act like either a push button or a
    toggle button (such as a checkbox) with appropriate traits listeners.

    Components which use this class can listen to the ``clicked`` event
    or listen to the ``checked`` state, depending on whether they want
    "push button" or "check box" style behaviour.

    Components may also want to listen to the ``down`` attribute to change
    the way that they are drawn in response to the mouse position, for example
    by highlighting the component.
    """

    # -------------------------------------------------------------------------
    # 'ButtonTool' interface
    # -------------------------------------------------------------------------

    #: Event fired when button is clicked
    clicked = Event

    #: Is the button toggled?
    checked = Bool(False)

    #: Is the mouse button pressed down in the clickable region
    down = Bool(False)

    #: whether or not the button can be pressed.
    enabled = Bool(True)

    #: whether or not the button can be toggled (eg. checkbox or radio button).
    togglable = Bool(False)

    def is_clickable(self, x, y):
        """ Is the (x,y) position in a region that responds to clicks.

        Used by the tool to determine when to start a click and when the
        button should be considered pressed down (this controls the state
        of the ``down`` trait).
        """
        return self.component.is_in(x, y)

    def click(self):
        """ Perform a click, toggling if needed, and firing the clicked event

        This doesn't change the state of the ``down`` trait.
        """
        if self.togglable:
            self.toggle()
        self.clicked = True

    def toggle(self):
        """ Toggle the state of the button.

        This does not fire the clicked event.

        Default is to invert the checked state, but subclasses could implement
        move complex cycling of toggle states.
        """
        self.checked = not self.checked

    # -------------------------------------------------------------------------
    # 'BaseTool' stateful event handlers
    # -------------------------------------------------------------------------

    def normal_left_down(self, event):
        if self.enabled and self.is_clickable(event.x, event.y):
            event.window.mouse_owner = self
            self.down = True
            self.event_state = "pressed"
            self.component.active_tool = self
            event.handled = True

    def pressed_mouse_move(self, event):
        self.down = self.is_clickable(event.x, event.y)

    def pressed_mouse_leave(self, event):
        self.down = False

    def pressed_mouse_enter(self, event):
        self.down = self.is_clickable(event.x, event.y)

    def pressed_left_up(self, event):
        if self.down:
            self.click()
            event.handled = True
        event.window.mouse_owner = None
        self.down = False
        self.event_state = "normal"
