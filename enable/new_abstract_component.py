""" Defines the AbstractComponent class """

# Enthought library imports
from traits.api import Any, Enum, Instance, List, Property

# Local relative imports
from enable_traits import coordinate_trait
from interactor import Interactor

class AbstractComponent(Interactor):
    """
    AbstractComponent is the primitive base class for Component.  It only
    requires the ability to handle events and render itself.  It supports
    being contained within a parent graphical object and defines methods
    to handle positions.  It does not necessarily have bounds, nor does it
    have any notion of viewports.
    """

    #------------------------------------------------------------------------
    # Positioning traits
    #------------------------------------------------------------------------

    # The position relative to the container.  If container is None, then
    # position will be set to (0,0).
    position = coordinate_trait

    # X-coordinate of our position
    x = Property

    # Y-coordinate of our position
    y = Property

    #------------------------------------------------------------------------
    # Object/containment hierarchy traits
    #------------------------------------------------------------------------

    # Our container object
    container = Any    # Instance("Container")

    # A reference to our top-level Enable Window
    window = Property   # Instance("Window")

    # The list of viewport that are viewing this component
    viewports = List(Instance("enable.Viewport"))

    #------------------------------------------------------------------------
    # Other public traits
    #------------------------------------------------------------------------

    # How this component should draw itself when draw() is called with
    # a mode of "default".  (If a draw mode is explicitly passed in to
    # draw(), then this is overruled.)
    # FIXME: Appears to be unused 5/3/6
    default_draw_mode = Enum("normal", "interactive")

    #------------------------------------------------------------------------
    # Private traits
    #------------------------------------------------------------------------

    # Shadow trait for self.window.  Only gets set if this is the top-level
    # enable component in a Window.
    _window = Any    # Instance("Window")

    #------------------------------------------------------------------------
    # Public concrete methods
    # (Subclasses should not override these; they provide an extension point
    # for mix-in classes.)
    #------------------------------------------------------------------------

    def __init__(self, **traits):
        # The only reason we need the constructor is to make sure our container
        # gets notified of our being added to it.
        if traits.has_key("container"):
            container = traits.pop("container")
            Interactor.__init__(self, **traits)
            container.add(self)
        else:
            Interactor.__init__(self, **traits)
        return

    def get_absolute_coords(self, *coords):
        """ Given coordinates relative to this component's origin, returns
        the "absolute" coordinates in the frame of the top-level parent
        Window enclosing this component's ancestor containers.

        Can be called in two ways:
            get_absolute_coords(x, y)
            get_absolute_coords( (x,y) )

        Returns a tuple (x,y) representing the new coordinates.
        """
        if self.container is not None:
            offset_x, offset_y = self.container.get_absolute_coords(*self.position)
        else:
            offset_x, offset_y = self.position
        return (offset_x + coords[0], offset_y + coords[1])

    def request_redraw(self):
        """
        Requests that the component redraw itself.  Usually this means asking
        its parent for a repaint.
        """
        for view in self.viewports:
            view.request_redraw()
        self._request_redraw()
        return


    #------------------------------------------------------------------------
    # Abstract public and protected methods that subclasses can override
    #------------------------------------------------------------------------

    def is_in(self, x, y):
        """
        Returns True if the point (x,y) is inside this component, False
        otherwise.  Even though AbstractComponents are not required to have
        bounds, they are still expected to be able to answer the question,
        "Does the point (x,y) lie within my region of interest?"  If so,
        then is_in() should return True.
        """
        raise NotImplementedError

    def _request_redraw(self):
        if self.container is not None:
            self.container.request_redraw()
        elif self._window:
            self._window.redraw()
        return


    #------------------------------------------------------------------------
    # Event handlers, getters & setters
    #------------------------------------------------------------------------

    def _container_changed(self, old, new):
        # We don't notify our container of this change b/c the
        # caller who changed our .container should take care of that.
        if new is None:
            self.position = [0,0]
        return

    def _position_changed(self):
        if self.container is not None:
            self.container._component_position_changed(self)
        return

    def _get_x(self):
        return self.position[0]

    def _set_x(self, val):
        self.position[0] = val
        return

    def _get_y(self):
        return self.position[1]

    def _set_y(self, val):
        self.position[1] = val
        return

    def _get_window(self, win):
        return self._window

    def _set_window(self, win):
        self._window = win
        return

    ### Persistence ###########################################################

    def __getstate__(self):
        state = super(AbstractComponent,self).__getstate__()
        for key in ['_window', 'viewports']:
            if state.has_key(key):
                del state[key]

        return state

# EOF
