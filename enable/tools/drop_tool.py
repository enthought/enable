#------------------------------------------------------------------------------
# Copyright (c) 2014, Enthought, Inc.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#------------------------------------------------------------------------------
""" Abstract base class for tools that handle drag and drop """

from __future__ import absolute_import, print_function, division

from enable.base_tool import BaseTool


class BaseDropTool(BaseTool):
    """ Abstract base class for tools that handle drag and drop """

    def normal_drag_over(self, event):
        """ Handle dragging over the component """
        try:
            result = self.accepts_drop(event.obj)
            self.component.window.set_drag_result(result)
        except Exception:
            self.component.window.set_drag_result("error")
            raise

    def normal_dropped_on(self, event):
        if self.accepts_drop(event.obj) != "none":
            self.handle_drop(event.obj)

    def accepts_drop(self, urls):
        """ Whether or not to accept the drag, and the type of drag

        The return value is either "none", if the drag is refused for the
        dragged object types, or one of "copy", "move", or "link".

        Subclasses should override this method.

        """
        raise NotImplementedError

    def handle_drop(self, urls):
        """ Handle objects being dropped on the component

        Subclasses should override this method.
        """
        raise NotImplementedError
