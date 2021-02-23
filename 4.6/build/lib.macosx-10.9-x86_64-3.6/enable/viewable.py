# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# Enthought library imports
from traits.api import HasTraits


class Viewable(HasTraits):
    """
    Mixin class for Components which want to support being rendered by
    multiple viewports.
    """

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    def request_redraw(self):
        # This overrides the default Component request_redraw by asking
        # all of the views to redraw themselves.
        return

    def draw(self, gc, view_bounds=None, mode="default"):
        if len(self.viewports) > 0:
            for view in self.viewports:
                view.draw(gc, view_bounds, mode)
        else:
            super(Viewable, self).draw(gc, view_bounds, mode)
