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
Defines the base class for all Chaco tools.  See docs/event_handling.txt for an
overview of how event handling works in Chaco.
"""


# Enthought library imports
from traits.api import Bool, Enum, Instance

# Local relative imports
from .component import Component
from .interactor import Interactor


class KeySpec(object):
    """
    Creates a key specification to facilitate tools interacting with the
    keyboard.  A tool can declare either a class attribute::

        magic_key = KeySpec("Right", "control", ignore=['shift'])

    or a trait::

        magic_key = Instance(KeySpec, args=("Right", "control"),
                             kw={'ignore': ['shift']})

    and then check to see if the key was pressed by calling::

        if self.magic_key.match(event):
            # do stuff...

    The names of the keys come from Enable, so both examples above
    are specifying the user pressing Ctrl + Right_arrow with Alt not pressed
    and Shift either pressed or not.
    """

    def __init__(self, key, *modifiers, **kwmods):
        """ Creates this key spec with the given modifiers. """
        self.key = key
        mods = set(m.lower() for m in modifiers)
        self.alt = "alt" in mods
        self.shift = "shift" in mods
        self.control = "control" in mods
        ignore = kwmods.get("ignore", [])
        self.ignore = set(m.lower() for m in ignore)

    def match(self, event):
        """
        Returns True if the given Enable key_pressed event matches this key
        specification.
        """
        return (
            (self.key == getattr(event, "character", None))
            and ("alt" in self.ignore or self.alt == event.alt_down)
            and (
                "control" in self.ignore or self.control == event.control_down
            )
            and ("shift" in self.ignore or self.shift == event.shift_down)
        )

    @classmethod
    def from_string(cls, s):
        """ Create a KeySpec from a string joined by '+' characters. """
        codes = s.split("+")
        key = codes[-1]
        modifiers = set(code.lower() for code in codes[:-1])
        ignore = set("alt", "shift", "control") - modifiers
        return cls(key, *modifiers, ignore=ignore)


class BaseTool(Interactor):
    """ The base class for Chaco tools.

    Tools are not Enable components, but they can draw.  They do not
    participate in layout, but are instead attached to a Component, which
    dispatches methods to the tool and calls the tools' draw() method.

    See docs/event_handling.txt for more information on how tools are
    structured.
    """

    # The component that this tool is attached to.
    component = Instance(Component)

    # Is this tool's visual representation visible?  For passive inspector-type
    # tools, this is a constant value set in the class definition;
    # for stateful or modal tools, the tool's listener sets this attribute.
    visible = Bool(False)

    # How the tool draws on top of its component.  This, in conjuction with a
    # a tool's status on the component, is used by the component to determine
    # how to render itself.  In general, the meanings of the draw modes are:
    #
    # normal:
    #     The appearance of part of the component is modified such that
    #     the component is redrawn even if it has not otherwise
    #     received any indication that its previous rendering is invalid.
    #     The tool controls its own drawing loop, and calls out to this
    #     tool after it is done drawing itself.
    # overlay:
    #     The component needs to be drawn, but can be drawn after all
    #     of the background and foreground elements in the component.
    #     Furthermore, the tool renders correctly regardless
    #     of how the component renders itself (e.g., via a cached image).
    #     The overlay gets full control of the rendering loop, and must
    #     explicitly call the component's _draw() method; otherwise the
    #     component does not render.
    # none:
    #     The tool does not have a visual representation that the component
    #     needs to render.
    draw_mode = Enum("none", "overlay", "normal")

    # ------------------------------------------------------------------------
    # Concrete methods
    # ------------------------------------------------------------------------

    def __init__(self, component=None, **kwtraits):
        if "component" in kwtraits:
            component = kwtraits["component"]
        super(BaseTool, self).__init__(**kwtraits)
        self.component = component

    def dispatch(self, event, suffix):
        """ Dispatches a mouse event based on the current event state.

        Overrides enable.Interactor.
        """
        self._dispatch_stateful_event(event, suffix)

    def _dispatch_stateful_event(self, event, suffix):
        # Override the default enable.Interactor behavior of automatically
        # setting the event.handled if a handler is found.  (Without this
        # level of manual control, we could never support multiple listeners.)
        handler = getattr(self, self.event_state + "_" + suffix, None)
        if handler is not None:
            handler(event)

    # ------------------------------------------------------------------------
    # Abstract methods that subclasses should implement
    # ------------------------------------------------------------------------

    def draw(self, gc, view_bounds=None):
        """ Draws this tool on a graphics context.

        It is assumed that the graphics context has a coordinate transform that
        matches the origin of its component. (For containers, this is just the
        origin; for components, it is the origin of their containers.)
        """
        pass

    def _activate(self):
        """ Called by a Component when this becomes the active tool.
        """
        pass

    def _deactivate(self):
        """ Called by a Component when this is no longer the active tool.
        """
        pass

    def deactivate(self, component=None):
        """ Handles this component no longer being the active tool.
        """
        # Compatibility with new AbstractController interface
        self._deactivate()
