#------------------------------------------------------------------------------
# Copyright (c) 2011, Enthought, Inc.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#------------------------------------------------------------------------------

from __future__ import absolute_import

import warnings

from traits.qt import QtCore

from ..toolkit_constants import key_names, pointer_names

BUTTON_NAME_MAP = {
    QtCore.Qt.LeftButton:   "left",
    QtCore.Qt.RightButton:  "right",
    QtCore.Qt.MidButton:    "middle",
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

key_symbols = [
    QtCore.Qt.Key_Plus,
    QtCore.Qt.Key_Backspace,
    QtCore.Qt.Key_Cancel,
    QtCore.Qt.Key_CapsLock,
    QtCore.Qt.Key_Clear,
    QtCore.Qt.Key_Control,
    QtCore.Qt.Key_Period,
    QtCore.Qt.Key_Delete,
    QtCore.Qt.Key_Slash,
    QtCore.Qt.Key_Down,
    QtCore.Qt.Key_End,
    QtCore.Qt.Key_Return,
    QtCore.Qt.Key_Enter,
    QtCore.Qt.Key_Escape,
    QtCore.Qt.Key_Execute,
    QtCore.Qt.Key_F1,
    QtCore.Qt.Key_F10,
    QtCore.Qt.Key_F11,
    QtCore.Qt.Key_F12,
    QtCore.Qt.Key_F13,
    QtCore.Qt.Key_F14,
    QtCore.Qt.Key_F15,
    QtCore.Qt.Key_F16,
    QtCore.Qt.Key_F17,
    QtCore.Qt.Key_F18,
    QtCore.Qt.Key_F19,
    QtCore.Qt.Key_F2,
    QtCore.Qt.Key_F20,
    QtCore.Qt.Key_F21,
    QtCore.Qt.Key_F22,
    QtCore.Qt.Key_F23,
    QtCore.Qt.Key_F24,
    QtCore.Qt.Key_F3,
    QtCore.Qt.Key_F4,
    QtCore.Qt.Key_F5,
    QtCore.Qt.Key_F6,
    QtCore.Qt.Key_F7,
    QtCore.Qt.Key_F8,
    QtCore.Qt.Key_F9,
    QtCore.Qt.Key_Help,
    QtCore.Qt.Key_Home,
    QtCore.Qt.Key_Insert,
    QtCore.Qt.Key_Left,
    QtCore.Qt.Key_Meta,
    QtCore.Qt.Key_Asterisk,
    QtCore.Qt.Key_NumLock,
    QtCore.Qt.Key_0,
    QtCore.Qt.Key_1,
    QtCore.Qt.Key_2,
    QtCore.Qt.Key_3,
    QtCore.Qt.Key_4,
    QtCore.Qt.Key_5,
    QtCore.Qt.Key_6,
    QtCore.Qt.Key_7,
    QtCore.Qt.Key_8,
    QtCore.Qt.Key_9,
    QtCore.Qt.Key_PageDown,
    QtCore.Qt.Key_PageUp,
    QtCore.Qt.Key_Pause,
    QtCore.Qt.Key_Print,
    QtCore.Qt.Key_Right,
    QtCore.Qt.Key_ScrollLock,
    QtCore.Qt.Key_Select,
    QtCore.Qt.Key_Shift,
    QtCore.Qt.Key_Minus,
    QtCore.Qt.Key_Tab,
    QtCore.Qt.Key_Up,
]

if len(key_symbols) != len(key_names):
    warnings.warn("The Qt4 toolkit backend keymap is out of sync!")

KEY_MAP = dict(zip(key_symbols, key_names))
