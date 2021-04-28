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
Example Application Support
===========================

This module provides a simple Pyface application that can be used by examples
in places where a DemoFrame is insufficient.

"""

from pyface.api import ApplicationWindow, GUI


class DemoApplication(ApplicationWindow):
    """ Simple Pyface application displaying a component.

    This application has the same interface as the DemoFrames from the
    example_support module, but the window is embedded in a full Pyface
    application.  This means that subclasses have the opportunity of
    adding Menus, Toolbars, and other similar features to the demo, where
    needed.

    """

    def _create_contents(self, parent):
        self.enable_win = self._create_window()
        return self.enable_win.control

    def _create_window(self):
        "Subclasses should override this method and return an enable.Window"
        raise NotImplementedError()

    @classmethod
    def demo_main(cls, **traits):
        """ Create the demo application and start the mainloop, if needed

        This should be called with appropriate arguments which will be passed
        to the class' constructor.

        """
        # get the Pyface GUI
        gui = GUI()

        # create the application's main window
        window = cls(**traits)
        window.open()

        # start the application
        # if there isn't already a running mainloop, this will block
        gui.start_event_loop()

        # if there is already a running mainloop (eg. in an IPython session),
        # return a reference to the window so that our caller can hold on to it
        return window


def demo_main(cls, **traits):
    """ Create the demo application and start the mainloop, if needed.

    This is a simple wrapper around `cls.demo_main` for compatibility with the
    `DemoFrame` implementation.

    This should be called with appropriate arguments which will be passed to
    the class' constructor.

    """
    cls.demo_main(**traits)
