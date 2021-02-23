# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

import warnings

import wx

from ..toolkit_constants import (
    pointer_names,
    key_names,
    mouse_wheel_axes_names,
)

DRAG_RESULTS_MAP = {
    "error": wx.DragError,
    "none": wx.DragNone,
    "copy": wx.DragCopy,
    "move": wx.DragMove,
    "link": wx.DragLink,
    "cancel": wx.DragCancel,
}

# Map from pointer shape name to pointer shapes:
pointer_shapes = [
    wx.CURSOR_ARROW,
    wx.CURSOR_ARROWWAIT,
    wx.CURSOR_BLANK,
    wx.CURSOR_BULLSEYE,
    wx.CURSOR_CHAR,
    wx.CURSOR_CROSS,
    wx.CURSOR_HAND,
    wx.CURSOR_IBEAM,
    wx.CURSOR_LEFT_BUTTON,
    wx.CURSOR_MAGNIFIER,
    wx.CURSOR_MIDDLE_BUTTON,
    wx.CURSOR_NO_ENTRY,
    wx.CURSOR_PAINT_BRUSH,
    wx.CURSOR_PENCIL,
    wx.CURSOR_POINT_LEFT,
    wx.CURSOR_POINT_RIGHT,
    wx.CURSOR_QUESTION_ARROW,
    wx.CURSOR_RIGHT_ARROW,
    wx.CURSOR_RIGHT_BUTTON,
    wx.CURSOR_SIZENS,
    wx.CURSOR_SIZENESW,
    wx.CURSOR_SIZENWSE,
    wx.CURSOR_SIZEWE,
    wx.CURSOR_SIZEWE,
    wx.CURSOR_SIZENS,
    wx.CURSOR_SIZENWSE,
    wx.CURSOR_SIZENESW,
    wx.CURSOR_SIZING,
    wx.CURSOR_SPRAYCAN,
    wx.CURSOR_WAIT,
    wx.CURSOR_WATCH,
]

if len(pointer_names) != len(pointer_shapes):
    warnings.warn("The WX toolkit backend pointer map is out of sync!")

POINTER_MAP = dict(zip(pointer_names, pointer_shapes))

# Map from wxPython special key names into Enable key names:
key_symbols = [
    wx.WXK_ADD,
    wx.WXK_BACK,
    wx.WXK_CANCEL,
    wx.WXK_CAPITAL,
    wx.WXK_CLEAR,
    wx.WXK_CONTROL,
    wx.WXK_DECIMAL,
    wx.WXK_DELETE,
    wx.WXK_DIVIDE,
    wx.WXK_DOWN,
    wx.WXK_END,
    wx.WXK_RETURN,
    wx.WXK_NUMPAD_ENTER,
    wx.WXK_ESCAPE,
    wx.WXK_EXECUTE,
    wx.WXK_F1,
    wx.WXK_F10,
    wx.WXK_F11,
    wx.WXK_F12,
    wx.WXK_F13,
    wx.WXK_F14,
    wx.WXK_F15,
    wx.WXK_F16,
    wx.WXK_F17,
    wx.WXK_F18,
    wx.WXK_F19,
    wx.WXK_F2,
    wx.WXK_F20,
    wx.WXK_F21,
    wx.WXK_F22,
    wx.WXK_F23,
    wx.WXK_F24,
    wx.WXK_F3,
    wx.WXK_F4,
    wx.WXK_F5,
    wx.WXK_F6,
    wx.WXK_F7,
    wx.WXK_F8,
    wx.WXK_F9,
    wx.WXK_HELP,
    wx.WXK_HOME,
    wx.WXK_INSERT,
    wx.WXK_LEFT,
    wx.WXK_MENU,
    wx.WXK_MULTIPLY,
    wx.WXK_NUMLOCK,
    wx.WXK_NUMPAD0,
    wx.WXK_NUMPAD1,
    wx.WXK_NUMPAD2,
    wx.WXK_NUMPAD3,
    wx.WXK_NUMPAD4,
    wx.WXK_NUMPAD5,
    wx.WXK_NUMPAD6,
    wx.WXK_NUMPAD7,
    wx.WXK_NUMPAD8,
    wx.WXK_NUMPAD9,
    wx.WXK_PAGEDOWN,  # Formerly: wx.WXK_NEXT
    wx.WXK_PAGEUP,  # Formerly: wx.WXK_PRIOR
    wx.WXK_PAUSE,
    wx.WXK_PRINT,
    wx.WXK_RIGHT,
    wx.WXK_SCROLL,
    wx.WXK_SELECT,
    wx.WXK_SHIFT,
    wx.WXK_SUBTRACT,
    wx.WXK_TAB,
    wx.WXK_UP,
    wx.WXK_ALT,
]


if len(key_symbols) != len(key_names):
    warnings.warn("The WX toolkit backend keymap is out of sync!")

KEY_MAP = dict(zip(key_symbols, key_names))

if wx.VERSION[:3] < (2, 9, 4):
    mouse_wheel_axes = [0, 1]
else:
    mouse_wheel_axes = [wx.MOUSE_WHEEL_VERTICAL, wx.MOUSE_WHEEL_HORIZONTAL]

if len(mouse_wheel_axes) != len(mouse_wheel_axes_names):
    warnings.warn("The WX toolkit backend mouse wheel axes are out of sync!")

MOUSE_WHEEL_AXIS_MAP = dict(zip(mouse_wheel_axes, mouse_wheel_axes_names))
