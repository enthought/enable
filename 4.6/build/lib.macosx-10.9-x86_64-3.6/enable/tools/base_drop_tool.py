# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Abstract base class for tools that handle drag and drop """

from traits.api import Enum

from enable.base_tool import BaseTool


class BaseDropTool(BaseTool):
    """ Abstract base class for tools that handle drag and drop
    """

    default_drag_result = Enum(
        "copy", "move", "link", "cancel", "error", "none"
    )

    def normal_drag_over(self, event):
        """ Handle dragging over the component
        """
        if event.handled:
            return
        try:
            result = self.get_drag_result((event.x, event.y), event.obj)
            if result is not None:
                event.window.set_drag_result(result)
                event.handled = True
        except Exception:
            event.window.set_drag_result("error")
            raise

    def normal_dropped_on(self, event):
        if event.handled:
            return
        position = (event.x, event.y)
        if self.accept_drop(position, event.obj):
            self.handle_drop(position, event.obj)
            event.handled = True

    def get_drag_result(self, position, obj):
        """ The type of drag that will happen

        By default, if the dragged objects are available this method calls
        accept_drop() and returns "none" if the result is False, otherwise
        it returns the value of default_drag_result.

        Parameters
        ----------
        position :
            The coordinates of the drag over event
        obj : any
            The object(s) being dragged, if available.  Some backends (such as
            Wx) may not be able to provide the object being dragged, in which
            case `obj` will be `None`.

        Returns
        -------
        Either None, if the drop should be ignored by this tool and not
        handled, or one of the keys of DRAG_RESULTS_MAP: "none", "copy, "move",
        "link", "cancel" or "error".
        """
        if obj is not None:
            # if we have the object, see if we can accept
            if not self.accept_drop(position, obj):
                return None

        return self.default_drag_result

    def accept_drop(self, position, obj):
        """ Whether or not to accept the drop

        Subclasses should override this method.

        Parameters
        ----------
        position :
            The coordinates of the drag over event
        obj : any
            The object(s) being dragged, if available.  Some backends (such as
            Wx) may not be able to provide the object being dragged, in which
            case `obj` will be `None`.

        Returns
        -------
        True if the drop should be accepted, False otherwise.
        """
        raise NotImplementedError

    def handle_drop(self, position, obj):
        """ Handle objects being dropped on the component

        Subclasses should override this method.

        Parameters
        ----------
        position :
            The coordinates of the drag over event
        obj : any
            The object(s) being dragged.
        """
        raise NotImplementedError
