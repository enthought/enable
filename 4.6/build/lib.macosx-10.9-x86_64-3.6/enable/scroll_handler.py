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
Interface for scroll handlers.
"""


class ScrollHandler:
    """ The interface for scroll handlers.

    A scroll handler handles the scroll events generated by scrollbar events
    in a Scrolled component.  By default, a Scrolled will serve as its own
    ScrollHandler.  In that role, Scrolled will merely move and clip the
    child component.

    If a component wishes to manage its own scrolling, it may do so, by
    implementing this interface and attaching itself as its parent's scroll
    manager.

    """

    def handle_vertical_scroll(self, position):
        """ Called when the vertical scroll position has changed.

        The position parameter will be the current position of the vertical
        scroll bar.

        """

        raise NotImplementedError

    def handle_horizontal_scroll(self, position):
        """ Called when the horizontal scroll position has changed.

        The position parameter will be the current position of the horizontal
        scroll bar.

        """

        raise NotImplementedError
