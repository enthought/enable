"""
Define a standard horizontal and vertical Enable scrollbar component that wraps
the standard WX one.
"""

# Major library imports
import wx
from types import ListType, TupleType

# Enthought Imports
from enthought.traits.api import Event, Property, Trait, TraitError, \
     Any, Str, Bool, Float, false, Int
from enthought.traits.ui.api import Group, View, Include

from enthought.enable2.component import Component

def valid_range(object, name, value):
    "Verify that a set of range values for a scrollbar is valid"
    try:
        if (type(value) in (TupleType, ListType)) and (len(value) == 4):
            low, high, page_size, line_size = value
            if high < low:
                low, high = high, low
            elif high == low:
                high = low + 1.0
            page_size = max(min(page_size, high - low), 0.0)
            line_size = max(min(line_size, page_size), 0.0)
            return (float(low), float(high), float(page_size), float(line_size))
    except:
        raise
    raise TraitError
valid_range.info = 'a (low,high,page_size,line_size) range tuple'

def valid_scroll_position(object, name, value):
    "Verify that a specified scroll bar position is valid"
    try:
        low, high, page_size, line_size = object.range
        return max(min(float(value), high - page_size), low)
    except:
        raise
    raise TraitError



class NativeScrollBar(Component):
    "An Enable scrollbar component that wraps/embeds the standard WX scrollbar"
    
    #------------------------------------------------------------------------
    # Public Traits
    #------------------------------------------------------------------------
    
    # The current position of the scroll bar.  This must be within the range
    # (self.low, self.high)
    scroll_position = Trait( 0.0, valid_scroll_position )
    
    # A tuple (low, high, page_size, line_size).  Can be accessed using
    # convenience properties (see below).
    range = Trait( ( 0.0, 100.0, 10.0, 1.0 ), valid_range )
    
    # The orientation of the scrollbar
    orientation = Trait("horizontal", "vertical")

    # Is y=0 at the top or bottom?
    origin = Trait('bottom', 'top')
    
    # Determines if the scroll bar should be visible and respond to events
    enabled = Bool(True)
    
    # The scroll increment associated with a single mouse wheel increment
    mouse_wheel_speed = Int(3)

    # Expose scroll_position, low, high, page_size as properties
    low = Property
    high = Property
    page_size = Property
    line_size = Property

    #------------------------------------------------------------------------
    # Private Traits
    #------------------------------------------------------------------------
    _control = Any(None)
    _clean = false
    _last_widget_x = Float(0)
    _last_widget_y = Float(0)
    _last_widget_height = Float(0)
    _list_widget_width = Float(0)

    #------------------------------------------------------------------------
    # Public Methods
    #------------------------------------------------------------------------

    def set_position(self, pos):
        """ Given a floating point number between 0.0 and 1.0 (inclusive),
            set the position of the scrollbar so that is scrolled approximately
            that percent.
        """
        if pos < 0 or pos > 1:
            raise ValueError, "pos must be between 0 and 1"
        if self._control:
            range = float(self._control.GetRange()-self._control.GetThumbSize())
            self._control.SetThumbPosition(int(round(pos*range)))
        return
    
    def destroy(self):
        """ Destroy the native widget associated with this component.
        """
        if self._control:
            self._control.Destroy()
        return
    
    #------------------------------------------------------------------------
    # Protected methods
    #------------------------------------------------------------------------
    
    def __del__(self):
        self.destroy()
        return

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        """Draw the component."""
        # To determine whether to actually redraw the component, we first check whether
        # we're clean or not.  If we are clean, we must additionally check whether we
        # have moved in wx coordinate space.  There's no easy way to get trait notification
        # on this because it depends on the entire position stack above us.  Therefore, we
        # compute this each time and redraw if it has changed.
        
        x_pos, y_pos = self.position
        x_size, y_size = self.bounds

        wx_xpos, wx_ypos = self.container.get_absolute_coords(x_pos, y_pos+y_size-1)
        
        # We have to do this flip_y business because wx and enable use opposite
        # coordinate systems, and enable defines the component's position as its
        # lower left corner, while wx defines it as the upper left corner.
        window = getattr(gc, "window", None)
        if window is None:
            return
        wx_ypos = window._flip_y(wx_ypos)

        if self._clean and \
               self._last_widget_x == wx_xpos and \
               self._last_widget_y == wx_ypos and \
               self._last_widget_width == x_size and \
               self._last_widget_height == y_size:
            return

        self._last_widget_x = wx_xpos
        self._last_widget_y = wx_ypos
        self._last_widget_width = x_size
        self._last_widget_height = y_size
        
        
        low, high, page_size, line_size = self.range
        if self.orientation == 'horizontal':
            wxstyle = wx.HORIZONTAL
        else:
            wxstyle = wx.VERTICAL
        tmp = self._enable_to_wx_spec(self.range + (self.scroll_position,))
        (wxpos, wxthumbsize, wxrange)  = tmp
        
        if not self._control:
            self._control = wx.ScrollBar(window.control, style=wxstyle)
            self._control.SetScrollbar(wxpos, wxthumbsize, wxrange, wxthumbsize, True)
            wx.EVT_SCROLL(self._control, self._wx_scroll_handler)
            wx.EVT_SET_FOCUS(self._control, self._yield_focus)
        
        # Ideally we would only SetPosition if the position change came from the
        # program rather than from the user.  Perhaps we should have a position_dirty
        # variable which is set by _scroll_position_changed or something like that.
        self._control.SetPosition(wx.Point(wx_xpos, wx_ypos))
        controlsize = self._control.GetSize()
        if x_size != controlsize[0] or y_size != controlsize[1]:
            self._control.SetSize(wx.Size(x_size, y_size))
        self._control.SetScrollbar(wxpos, wxthumbsize, wxrange, wxthumbsize, True)

        self._clean = True
        return
    
    def _yield_focus(self, event):
        """ Yields focus to our window, when we acquire focus via user interaction. """
        window = event.GetWindow()
        if window:
            window.SetFocus()
        return
    
    def _mouse_wheel_changed(self, event):
        event.handled  = True
        self.scroll_position -= (event.mouse_wheel * self.range[3] * self.mouse_wheel_speed)
        return

    def _scroll_position_changed(self):
        self._clean = False
        self.request_redraw()
        return
    
    def _range_changed(self):
        low, high, page_size, line_size = self.range
        self.scroll_position = max(min(self.scroll_position, high-page_size), low)
        self._clean = False
        self.request_redraw()
        return

    def _range_items_changed(self):
        self._range_changed()
        return

    def _wx_scroll_handler(self, event):
        """Handle wx scroll events"""
        #If the user moved the scrollbar, set the scroll position, but don't
        #tell wx to move the scrollbar.  Doing so causes jerkiness
        self.scroll_position = self._wx_to_enable_pos(self._control.GetThumbPosition())
        self._clean = True
        return
        
    def _enable_to_wx_spec(self, enable_spec):
        """Return the WX equivalent of an enable scroll bar specification
        From a tuple of (low, high, page_size, line_size, position),
        return (position, thumbsize, range)"""
        low, high, page_size, line_size, position = enable_spec
        if self.origin == 'bottom' and self.orientation == 'vertical':
            position = (high-page_size)-position
        if line_size == 0.0:
            return (0,high-low,high-low)
        else:
            return map(int, ((position-low)/line_size, page_size/line_size, (high-low)/line_size))

    def _wx_to_enable_pos(self, pos):
        """Translate the position that the Wx scrollbar returns into the position we store
        internally.  The difference is that we have a high and a low and a line size, while
        wx assumes low is 0 and line size is 1."""
        low, high, page_size, line_size = self.range
        enablepos = pos*line_size+low
        #If we're a veritcal scrollbar with a bottom origin, flip
        #the coordinates, since in WX the origin is always the top.
        if self.origin == 'bottom' and self.orientation == 'vertical':
            enablepos = (high-page_size)-enablepos
        return enablepos


    #------------------------------------------------------------------------
    # Property getters and setters
    #------------------------------------------------------------------------

    def _get_low(self):
        return self.range[0]
        
    def _set_low(self, low):
        ignore, high, page_size, line_size = self.range
        self._clean = False
        self.range =(low, high, page_size, line_size)
        
    def _get_high(self):
        return self.range[1]
        
    def _set_high(self, high):
        low, ignore, page_size, line_size = self.range
        self._clean = False
        self.range =(low, high, page_size, line_size)
        
    def _get_page_size(self):
        return self.range[2]
        
    def _set_page_size(self, page_size):
        low, high, ignore, line_size = self.range
        self._clean = False
        self.range =(low, high, page_size, line_size)
        
    def _get_line_size(self):
        return self.range[3]
        
    def _set_line_size(self, line_size):
        low, high, page_size, ignore = self.range
        self._clean = False
        self.range =(low, high, page_size, line_size)


# EOF
