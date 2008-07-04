
# Enthought library imports
from enthought.traits.api import HasTraits


class Viewable(HasTraits):
    """
    Mixin class for Components which want to support being rendered by
    multiple viewports.
    """

    #------------------------------------------------------------------------
    # Public methods
    #------------------------------------------------------------------------
    
    def request_redraw(self):
        # This overrides the default Component request_redraw by asking
        # all of the views to redraw themselves.
        print "request_redraw on viewable"
        return
    
    def draw(self, gc, view_bounds=None, mode="default"):
        if len(self.viewports) > 0:
            for view in self.viewports:
                view.draw(gc, view_bounds, mode)
        else:
            super(Viewable, self).draw(gc, view_bounds, mode)
        return

# EOF
