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

from pyface.qt import QtCore

from ..toolkit_constants import (
    key_names, mouse_wheel_axes_names, pointer_names
)

DRAG_RESULTS_MAP = {
    "error": QtCore.Qt.IgnoreAction,
    "none": QtCore.Qt.IgnoreAction,
    "copy": QtCore.Qt.CopyAction,
    "move": QtCore.Qt.MoveAction,
    "link": QtCore.Qt.LinkAction,
    "cancel": QtCore.Qt.IgnoreAction,
}

BUTTON_NAME_MAP = {
    QtCore.Qt.LeftButton: "left",
    QtCore.Qt.RightButton: "right",
    QtCore.Qt.MidButton: "middle",
    QtCore.Qt.XButton1: "back",
    QtCore.Qt.XButton2: "forward",
    QtCore.Qt.NoButton: "none",
}

# TODO: Create bitmap cursor for the following:
#   arrow wait
#   bullseye
#   char
#   magnifier
#   paint brush
#   pencil
#   point left
#   point right
#   right arrow
#   spray can

pointer_shapes = [
    QtCore.Qt.ArrowCursor,
    QtCore.Qt.BusyCursor,
    QtCore.Qt.BlankCursor,
    QtCore.Qt.CrossCursor,
    QtCore.Qt.IBeamCursor,
    QtCore.Qt.CrossCursor,
    QtCore.Qt.PointingHandCursor,
    QtCore.Qt.IBeamCursor,
    QtCore.Qt.ArrowCursor,
    QtCore.Qt.CrossCursor,
    QtCore.Qt.ArrowCursor,
    QtCore.Qt.ForbiddenCursor,
    QtCore.Qt.ArrowCursor,
    QtCore.Qt.CrossCursor,
    QtCore.Qt.ArrowCursor,
    QtCore.Qt.ArrowCursor,
    QtCore.Qt.WhatsThisCursor,
    QtCore.Qt.ArrowCursor,
    QtCore.Qt.ArrowCursor,
    QtCore.Qt.SizeVerCursor,
    QtCore.Qt.SizeBDiagCursor,
    QtCore.Qt.SizeFDiagCursor,
    QtCore.Qt.SizeHorCursor,
    QtCore.Qt.SizeHorCursor,
    QtCore.Qt.SizeVerCursor,
    QtCore.Qt.SizeFDiagCursor,
    QtCore.Qt.SizeBDiagCursor,
    QtCore.Qt.SizeAllCursor,
    QtCore.Qt.CrossCursor,
    QtCore.Qt.WaitCursor,
    QtCore.Qt.BusyCursor,
]

if len(pointer_names) != len(pointer_shapes):
    warnings.warn("The Qt4 toolkit backend pointer map is out of sync!")

POINTER_MAP = dict(zip(pointer_names, pointer_shapes))

KEY_MAP = {
    QtCore.Qt.Key_Backspace: "Backspace",
    QtCore.Qt.Key_Cancel: "Cancel",
    QtCore.Qt.Key_CapsLock: "Capital",
    QtCore.Qt.Key_Clear: "Clear",
    QtCore.Qt.Key_Control: "Control",
    QtCore.Qt.Key_Delete: "Delete",
    QtCore.Qt.Key_Down: "Down",
    QtCore.Qt.Key_End: "End",
    QtCore.Qt.Key_Return: "Enter",
    QtCore.Qt.Key_Enter: "Enter",
    QtCore.Qt.Key_Escape: "Esc",
    QtCore.Qt.Key_Execute: "Execute",
    QtCore.Qt.Key_F1: "F1",
    QtCore.Qt.Key_F10: "F10",
    QtCore.Qt.Key_F11: "F11",
    QtCore.Qt.Key_F12: "F12",
    QtCore.Qt.Key_F13: "F13",
    QtCore.Qt.Key_F14: "F14",
    QtCore.Qt.Key_F15: "F15",
    QtCore.Qt.Key_F16: "F16",
    QtCore.Qt.Key_F17: "F17",
    QtCore.Qt.Key_F18: "F18",
    QtCore.Qt.Key_F19: "F19",
    QtCore.Qt.Key_F2: "F2",
    QtCore.Qt.Key_F20: "F20",
    QtCore.Qt.Key_F21: "F21",
    QtCore.Qt.Key_F22: "F22",
    QtCore.Qt.Key_F23: "F23",
    QtCore.Qt.Key_F24: "F24",
    QtCore.Qt.Key_F3: "F3",
    QtCore.Qt.Key_F4: "F4",
    QtCore.Qt.Key_F5: "F5",
    QtCore.Qt.Key_F6: "F6",
    QtCore.Qt.Key_F7: "F7",
    QtCore.Qt.Key_F8: "F8",
    QtCore.Qt.Key_F9: "F9",
    QtCore.Qt.Key_Help: "Help",
    QtCore.Qt.Key_Home: "Home",
    QtCore.Qt.Key_Insert: "Insert",
    QtCore.Qt.Key_Left: "Left",
    QtCore.Qt.Key_Meta: "Menu",
    QtCore.Qt.Key_Asterisk: "Multiply",
    QtCore.Qt.Key_NumLock: "Num Lock",
    QtCore.Qt.Key_PageDown: "Page Down",
    QtCore.Qt.Key_PageUp: "Page Up",
    QtCore.Qt.Key_Pause: "Pause",
    QtCore.Qt.Key_Print: "Print",
    QtCore.Qt.Key_Right: "Right",
    QtCore.Qt.Key_ScrollLock: "Scroll Lock",
    QtCore.Qt.Key_Select: "Select",
    QtCore.Qt.Key_Shift: "Shift",
    QtCore.Qt.Key_Tab: "Tab",
    QtCore.Qt.Key_Up: "Up",
    QtCore.Qt.Key_Alt: "Alt",
}

# Add all of the other keys registered by Qt.
# This should work for both PySide and PyQt4.
for enum_name in dir(QtCore.Qt):
    if enum_name.startswith("Key_"):
        enum = getattr(QtCore.Qt, enum_name)
        # Ignore everything in latin-1 as we just want the unichr() conversion.
        if enum <= 255 or enum in KEY_MAP:
            continue
        key_name = enum_name[len("Key_"):]
        KEY_MAP[enum] = key_name

# set up mouse wheel axes constants
mouse_wheel_axes = [QtCore.Qt.Vertical, QtCore.Qt.Horizontal]
MOUSE_WHEEL_AXIS_MAP = dict(zip(mouse_wheel_axes, mouse_wheel_axes_names))
