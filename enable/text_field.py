# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# Standard library imports
from math import floor, sqrt

# Enthought library imports
from traits.api import (
    Any, Bool, DelegatesTo, Event, Instance, Int, List, Property
)

# Local, relative imports
from .component import Component
from .font_metrics_provider import font_metrics_provider
from .text_field_style import TextFieldStyle

StyleDelegate = DelegatesTo("_style")


class TextField(Component):
    """ A basic text entry field for Enable.
        fixme: Requires monospaced fonts.
    """

    # ------------------------------------------------------------------------
    # Public traits
    # ------------------------------------------------------------------------

    # The text to be edited
    text = Property(depends_on=["_text_changed"])

    # Events that get fired on certain keypresses
    accept = Event
    cancel = Event

    # Are multiple lines of text allowed?
    multiline = Bool(False)

    # The object to use to measure text extents
    metrics = Any
    char_w = Any
    char_h = Any

    # Is this text field editable?
    can_edit = Bool(True)

    # ------------------------------------------------------------------------
    # Delegates for style
    # ------------------------------------------------------------------------

    text_color = StyleDelegate
    font = StyleDelegate
    line_spacing = StyleDelegate
    text_offset = StyleDelegate
    cursor_color = StyleDelegate
    cursor_width = StyleDelegate
    border_visible = StyleDelegate
    border_color = StyleDelegate
    bgcolor = StyleDelegate

    # ------------------------------------------------------------------------
    # Protected traits
    # ------------------------------------------------------------------------

    # The style information used in drawing
    _style = Instance(TextFieldStyle, ())

    # The max width/height of the displayed text in characters
    _text_width = Property(
        depends_on=["_style", "height"], cached="_height_cache"
    )
    _text_height = Property(
        depends_on=["_style", "width"], cached="_width_cache"
    )

    # The x-y position of the cursor in the text
    _cursor_pos = List(Int)
    _old_cursor_pos = List(Int)
    _desired_cursor_x = Int

    # The text as an x-y grid, the shadow property for 'text'
    _text = List(List)
    _text_changed = Event

    # The text that is actually displayed in the editor, and its shadow values
    _draw_text = Property
    __draw_text = List(List)
    _draw_text_xstart = Int
    _draw_text_ystart = Int

    # Whether or not to draw the cursor (is mouse over box?)
    _draw_cursor = Bool(False)

    # fixme: Shouldn't traits initialize these on its own?
    # fixme again: I moved these out of __init__ because they weren't
    # accessible from the _get__text_height and _get__text_width methods.
    # Not sure if this is the right fix (dmartin)
    _width_cache = None
    _height_cache = None

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    def __init__(self, **traits):
        # This will be overriden if 'text' is provided as a trait, but it
        # must be initialized if not
        self._text = [[]]

        # Initialize internal tracking variables
        self.reset()

        super(TextField, self).__init__(**traits)

        if self.metrics is None:
            self.metrics = font_metrics_provider()

        # Initialize border/bg colors
        self.__style_changed()

        # If this can't be editted and no width has been set, make sure
        # that is wide enough to display the text.
        if not self.can_edit and self.width == 0:
            x, y, width, height = self.metrics.get_text_extent(self.text)
            offset = 2 * self._style.text_offset
            self.width = width + offset

    # ------------------------------------------------------------------------
    # Interactor interface
    # ------------------------------------------------------------------------

    def normal_mouse_enter(self, event):
        if not self.can_edit:
            return

        event.window.set_pointer("ibeam")
        self.request_redraw()
        event.handled = True

    def normal_mouse_leave(self, event):
        if not self.can_edit:
            return

        event.window.set_pointer("arrow")
        self.request_redraw()
        event.handled = True

    def normal_left_down(self, event):
        if not self.can_edit:
            return

        self.event_state = "cursor"
        self._acquire_focus(event.window)
        event.handled = True

        # Transform pixel coordinates to text coordinates
        char_width, char_height = self.metrics.get_text_extent("T")[2:]
        char_height += self._style.line_spacing
        event_x = event.x - self.x - self._style.text_offset
        event_y = self.y2 - event.y - self._style.text_offset
        if self.multiline:
            y = int(round(event_y / char_height)) - 1
        else:
            y = 0
        x = int(round(event_x / char_width))

        # Clip x and y so that they are with text bounds, then place the cursor
        y = min(max(y, 0), len(self.__draw_text) - 1)
        x = min(max(x, 0), len(self.__draw_text[y]))
        self._old_cursor_pos = self._cursor_pos
        self._cursor_pos = [
            self._draw_text_ystart + y,
            self._draw_text_xstart + x,
        ]

    def cursor_left_up(self, event):
        if not self.can_edit:
            return

        # Reset event state
        self.event_state = "normal"
        event.handled = True
        self.request_redraw()

    def normal_character(self, event):
        "Actual text that we want to add to the buffer as-is."
        # XXX need to filter unprintables that are not handled in key_pressed
        if not self.can_edit:
            return

        # Save for bookkeeping purposes
        self._old_cursor_pos = self._cursor_pos

        y, x = self._cursor_pos
        self._text[y].insert(x, event.character)
        self._cursor_pos[1] += 1
        self._desired_cursor_x = self._cursor_pos[1]
        self._text_changed = True

        event.handled = True
        self.invalidate_draw()
        self.request_redraw()

    def normal_key_pressed(self, event):
        "Special character handling"
        if not self.can_edit:
            return

        # Save for bookkeeping purposes
        self._old_cursor_pos = self._cursor_pos

        if event.character == "Backspace":
            # Normal delete
            if self._cursor_pos[1] > 0:
                del self._text[self._cursor_pos[0]][self._cursor_pos[1] - 1]
                self._cursor_pos[1] -= 1
                self._desired_cursor_x = self._cursor_pos[1]
                self._text_changed = True
            # Delete at the beginning of a line
            elif self._cursor_pos[0] - 1 >= 0:
                index = self._cursor_pos[0] - 1
                old_line_len = len(self._text[index])
                self._text[index] += self._text[index + 1]
                del self._text[index + 1]
                del self.__draw_text[index + 1 - self._draw_text_xstart]
                self._cursor_pos[0] -= 1
                self._cursor_pos[1] = old_line_len
                self._desired_cursor_x = self._cursor_pos[1]
                self._text_changed = True
        elif event.character == "Delete":
            # Normal delete
            if self._cursor_pos[1] < len(self._text[self._cursor_pos[0]]):
                del self._text[self._cursor_pos[0]][self._cursor_pos[1]]
                self._desired_cursor_x = self._cursor_pos[1]
                self._text_changed = True
            # Delete at the end of a line
            elif self._cursor_pos[0] + 1 < len(self._text):
                index = self._cursor_pos[0]
                old_line_len = len(self._text[index])
                self._text[index] += self._text[index + 1]
                del self._text[index + 1]
                del self.__draw_text[index + 1 - self._draw_text_xstart]
                self._desired_cursor_x = self._cursor_pos[1]
                self._text_changed = True

        # Cursor movement
        elif event.character == "Left":
            self._cursor_pos[1] -= 1
            if self._cursor_pos[1] < 0:
                self._cursor_pos[0] -= 1
                if self._cursor_pos[0] < 0:
                    self._cursor_pos = [0, 0]
                else:
                    self._cursor_pos[1] = len(self._text[self._cursor_pos[0]])
            self._desired_cursor_x = self._cursor_pos[1]
        elif event.character == "Right":
            self._cursor_pos[1] += 1
            if self._cursor_pos[1] > len(self._text[self._cursor_pos[0]]):
                self._cursor_pos[0] += 1
                if self._cursor_pos[0] > len(self._text) - 1:
                    self._cursor_pos[0] -= 1
                    self._cursor_pos[1] -= 1
                else:
                    self._cursor_pos[1] = 0
            self._desired_cursor_x = self._cursor_pos[1]
        elif event.character == "Up":
            self._cursor_pos[0] -= 1
            if self._cursor_pos[0] < 0:
                self._cursor_pos[0] = 0
            else:
                self._cursor_pos[1] = min(
                    len(self._text[self._cursor_pos[0]]),
                    self._desired_cursor_x,
                )
        elif event.character == "Down":
            self._cursor_pos[0] += 1
            if self._cursor_pos[0] >= len(self._text):
                self._cursor_pos[0] = len(self._text) - 1
            else:
                self._cursor_pos[1] = min(
                    len(self._text[self._cursor_pos[0]]),
                    self._desired_cursor_x,
                )
        elif event.character == "Home":
            self._cursor_pos[1] = 0
            self._desired_cursor_x = self._cursor_pos[1]
        elif event.character == "End":
            self._cursor_pos[1] = len(self._text[self._cursor_pos[0]])
            self._desired_cursor_x = self._cursor_pos[1]

        # Special characters
        elif event.character == "Tab":
            y, x = self._cursor_pos
            self._text[y] = self._text[y][:x] + [" "] * 4 + self._text[y][x:]
            self._cursor_pos[1] += 4
            self._desired_cursor_x = self._cursor_pos[1]
            self._text_changed = True
        elif event.character == "Enter":
            if self.multiline:
                line = self._cursor_pos[0]
                self._text.insert(
                    line + 1, self._text[line][self._cursor_pos[1]:]
                )
                self._text[line] = self._text[line][: self._cursor_pos[1]]
                self._cursor_pos[0] += 1
                self._cursor_pos[1] = 0
                self._desired_cursor_x = self._cursor_pos[1]
                self._text_changed = True
            else:
                self.accept = event
        elif event.character == "Escape":
            self.cancel = event
        elif len(event.character) == 1:
            # XXX normal keypress, so let it go through
            return

        event.handled = True
        self.invalidate_draw()
        self.request_redraw()

    # ------------------------------------------------------------------------
    # Component interface
    # ------------------------------------------------------------------------

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        with gc:
            # Draw the text
            gc.set_font(self._style.font)
            gc.set_fill_color(self._style.text_color)
            char_w, char_h = self.metrics.get_text_extent("T")[2:4]
            char_h += self._style.line_spacing
            lines = ["".join(ln) for ln in self._draw_text]
            for i, line in enumerate(lines):
                x = self.x + self._style.text_offset
                if i > 0:
                    y_offset = (i + 1) * char_h - self._style.line_spacing
                else:
                    y_offset = char_h - self._style.line_spacing
                y = self.y2 - y_offset - self._style.text_offset

                # Show text at the same scale as the graphics context
                ctm = gc.get_ctm()
                if hasattr(ctm, "__len__") and len(ctm) == 6:
                    scale = sqrt(
                        (ctm[0] + ctm[1]) * (ctm[0] + ctm[1]) / 2.0
                        + (ctm[2] + ctm[3]) * (ctm[2] + ctm[3]) / 2.0
                    )
                elif hasattr(gc, "get_ctm_scale"):
                    scale = gc.get_ctm_scale()
                else:
                    raise RuntimeError("Unable to get scale from GC.")
                x *= scale
                y *= scale
                gc.show_text_at_point(line, x, y)

            if self._draw_cursor:
                j, i = self._cursor_pos
                j -= self._draw_text_ystart
                i -= self._draw_text_xstart
                x_offset = self.metrics.get_text_extent(lines[j][:i])[2]
                y_offset = char_h * j
                y = self.y2 - y_offset - self._style.text_offset
                if not self.multiline:
                    char_h -= float(self._style.line_spacing) * 0.5

                gc.set_line_width(self._style.cursor_width)
                gc.set_stroke_color(self._style.cursor_color)
                gc.begin_path()
                x_position = self.x + x_offset + self._style.text_offset
                gc.move_to(x_position, y)
                gc.line_to(x_position, y - char_h)

                gc.stroke_path()

    # ------------------------------------------------------------------------
    # TextField interface
    # ------------------------------------------------------------------------

    def reset(self):
        """ Resets the text field. This involes reseting cursor position, text
        position, etc.
        """
        self._cursor_pos = [0, 0]
        self._old_cursor_pos = [0, 0]
        self.__draw_text = [[]]

    def _scroll_horz(self, num):
        """ Horizontally scrolls all the text that is being drawn by 'num'
        characters. If num is negative, scrolls left. If num is positive,
        scrolls right.
        """
        self._draw_text_xstart += num
        self._realign_horz()

    def _realign_horz(self):
        """ Realign all the text being drawn such that the first character
        being drawn in each line is the one at index '_draw_text_xstart.'
        """
        for i in range(len(self.__draw_text)):
            line = self._text[self._draw_text_ystart + i]
            self.__draw_text[i] = self._clip_line(line, self._draw_text_xstart)

    def _scroll_vert(self, num):
        """ Vertically scrolls all the text that is being drawn by 'num' lines.
        If num is negative, scrolls up. If num is positive, scrolls down.
        """
        x, y = self._draw_text_xstart, self._draw_text_ystart
        if num < 0:
            self.__draw_text = self.__draw_text[:num]
            lines = [
                self._clip_line(line, x) for line in self._text[y + num:y]
            ]
            self.__draw_text = lines + self.__draw_text
        elif num > 0:
            self.__draw_text = self.__draw_text[num:]
            y += self._text_height
            lines = [
                self._clip_line(line, x) for line in self._text[y:y + num]
            ]
            self.__draw_text.extend(lines)
        self._draw_text_ystart += num

    def _clip_line(self, text, index, start=True):
        """ Return 'text' clipped beginning at 'index' if 'start' is True or
        ending at 'index' if 'start' is False.
        """
        box_width = self.width - 2 * self._style.text_offset
        total_width = 0.0
        end_index = 1
        for t in text:
            w, h = self.metrics.get_text_extent(t)[2:4]
            total_width = total_width + w
            if total_width <= box_width:
                end_index = end_index + 1
            else:
                break

        if start:
            return text[index:min(index + end_index - 1, len(text))]
        else:
            return text[max(0, index - end_index):index]

    def _refresh_viewed_line(self, line):
        """ Updates the appropriate line in __draw_text with the text at 'line'
        """
        new_text = self._clip_line(self._text[line], self._draw_text_xstart)
        index = line - self._draw_text_ystart
        if index == len(self.__draw_text):
            self.__draw_text.append(new_text)
        else:
            self.__draw_text[index] = new_text

    def _acquire_focus(self, window):
        self._draw_cursor = True
        window.focus_owner = self
        window.on_trait_change(self._focus_owner_changed, "focus_owner")
        self.request_redraw()

    def _focus_owner_changed(self, obj, name, old, new):
        if old == self and new != self:
            obj.on_trait_change(
                self._focus_owner_changed, "focus_owner", remove=True
            )
        self._draw_cursor = False
        self.request_redraw()

    # ------------------------------------------------------------------------
    # Property getters/setters and trait event handlers
    # ------------------------------------------------------------------------

    def _get_text(self):
        return "\n".join(["".join(line) for line in self._text])

    def _set_text(self, val):
        if val == "":
            self._text = [[]]
        else:
            self._text = [list(line) for line in val.splitlines()]
        self.reset()
        self.request_redraw()

    def _get__draw_text(self):
        # Rebuilding from scratch
        if self.__draw_text == [[]]:
            if self.multiline:
                self.__draw_text = []
                self._draw_text_xstart, self._draw_text_ystart = 0, 0
                end = min(len(self._text), self._text_height)
                for i in range(self._draw_text_ystart, end):
                    line = self._clip_line(self._text[i], 0)
                    self.__draw_text.append(line)
            else:
                self.__draw_text = [self._clip_line(self._text[0], 0)]

        # Updating only the things that need updating
        else:
            # Scroll if necessary depending on where cursor moved
            # Adjust up
            if self._cursor_pos[0] < self._draw_text_ystart:
                self._scroll_vert(-1)

            # Adjust down
            elif (self._cursor_pos[0] - self._draw_text_ystart
                    >= self._text_height):
                self._scroll_vert(1)

            # Adjust left
            line = self._text[self._cursor_pos[0]]
            chars_before_start = len(line[: self._draw_text_xstart])
            chars_after_start = len(line[self._draw_text_xstart:])
            if self._cursor_pos[1] < self._draw_text_xstart:
                if chars_before_start <= self._text_width:
                    self._draw_text_xstart = 0
                    self._realign_horz()
                else:
                    self._scroll_horz(-self._text_width)
            if (self._draw_text_xstart > 0
                    and chars_after_start + 1 < self._text_width):
                self._scroll_horz(-1)

            # Adjust right
            num_chars = self._cursor_pos[1] - self._draw_text_xstart
            if num_chars >= self._text_width:
                self._scroll_horz(num_chars - self._text_width + 1)

            # Replace text at cursor location
            if self._old_cursor_pos[0] < self._cursor_pos[0]:
                # A line has been created by an enter event
                self._refresh_viewed_line(self._old_cursor_pos[0])
            self._refresh_viewed_line(self._cursor_pos[0])

        return self.__draw_text

    def _get__text_width(self):
        if self._width_cache is None:
            if self.metrics is not None:
                char_width = self.metrics.get_text_extent("T")[2]
                width = self.width - 2 * self._style.text_offset
                self._width_cache = int(floor(width / char_width))
        return self._width_cache

    def _get__text_height(self):
        if self.multiline:
            if self._height_cache is None:
                if self.metrics is not None:
                    char_height = self.metrics.get_text_extent("T")[3]
                    height = self.height - 2 * self._style.text_offset
                    line_height = char_height + self._style.line_spacing
                    self._height_cache = int(floor(height / line_height))
            return self._height_cache
        else:
            return 1

    def __style_changed(self):
        """ Bg/border color is inherited from the style, so update it when the
        style changes. The height of a line also depends on style.
        """
        self.bgcolor = self._style.bgcolor
        self.border_visible = self._style.border_visible
        self.border_color = self._style.border_color

        self.metrics.set_font(self._style.font)
        # FIXME!!  The height being passed in gets over-written here
        # if not self.multiline:
        #    self.height = (self.metrics.get_text_extent("T")[3] +
        #                   self._style.text_offset*2)

        self.request_redraw()
