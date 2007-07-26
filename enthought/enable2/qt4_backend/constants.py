
from PyQt4.Qt import Qt

BUTTON_NAME_MAP = {
    Qt.LeftButton: "left",
    Qt.RightButton: "right",
    Qt.MiddleButton: "middle",
}

# TODO: Create bitmap cursor for the following:
#   right arrow
#   bullseye
#   char
#   magnifier
#   paint brush
#   pencil
#   point left
#   point right
#   spray can
#   arrow wait

POINTER_MAP = {
   'arrow':            Qt.ArrowCursor,
   'right arrow':      Qt.ArrowCursor,
   'blank':            Qt.BlankCursor,
   'bullseye':         Qt.CrossCursor,
   'char':             Qt.IBeamCursor,
   'cross':            Qt.CrossCursor,
   'hand':             Qt.PointingHandCursor,
   'ibeam':            Qt.IBeamCursor,
   'left button':      Qt.ArrowCursor,
   'magnifier':        Qt.CrossCursor,
   'middle button':    Qt.ArrowCursor,
   'no entry':         Qt.ForbiddenCursor,
   'paint brush':      Qt.ArrowCursor,
   'pencil':           Qt.CrossCursor,
   'point left':       Qt.ArrowCursor,
   'point right':      Qt.ArrowCursor,
   'question arrow':   Qt.WhatsThisCursor,
   'right button':     Qt.ArrowCursor,
   'size top':         Qt.SizeVerCursor,
   'size bottom':      Qt.SizeVerCursor,
   'size left':        Qt.SizeHorCursor,
   'size right':       Qt.SizeHorCursor,
   'size top right':   Qt.SizeBDiagCursor,
   'size bottom left': Qt.SizeBDiagCursor,
   'size top left':    Qt.SizeFDiagCursor,
   'size bottom right':Qt.SizeFDiagCursor,
   'sizing':           Qt.SizeAllCursor,
   'spray can':        Qt.CrossCursor,
   'wait':             Qt.WaitCursor,
   'watch':            Qt.BusyCursor,
   'arrow wait':       Qt.BusyCursor,
}

KEY_MAP = {
    Qt.Key_Escape: "Esc",
    Qt.Key_Tab: "Tab",
    Qt.Key_Backtab: "Backtab",
    Qt.Key_Backspace: "Backspace",
    Qt.Key_Return: "Enter",
    Qt.Key_Enter: "Enter",
    Qt.Key_Insert: "Insert",
    Qt.Key_Delete: "Delete",
    Qt.Key_Pause: "Pause",
    Qt.Key_Print: "Print",
    Qt.Key_SysReq: "Sysreq",
    Qt.Key_Clear: "Clear",
    Qt.Key_Home: "Home",
    Qt.Key_End: "End",
    Qt.Key_Left: "Left",
    Qt.Key_Up: "Up",
    Qt.Key_Right: "Right",
    Qt.Key_Down: "Down",
    Qt.Key_PageUp: "Page Up",
    Qt.Key_PageDown: "Page Down",
    Qt.Key_Shift: "Shift",
    Qt.Key_Control: "Control",
    Qt.Key_Meta: "Meta",
    Qt.Key_Alt: "Alt",
    Qt.Key_AltGr: "Alt",
    Qt.Key_CapsLock: "Caps Lock",
    Qt.Key_NumLock: "Num Lock",
    Qt.Key_ScrollLock: "Scroll Lock",
    Qt.Key_F1:  "F1",
    Qt.Key_F2:  "F2",
    Qt.Key_F3:  "F3",
    Qt.Key_F4:  "F4",
    Qt.Key_F5:  "F5",
    Qt.Key_F6:  "F6",
    Qt.Key_F7:  "F7",
    Qt.Key_F8:  "F8",
    Qt.Key_F9:  "F9",
    Qt.Key_F10:  "F10",
    Qt.Key_F11:  "F11",
    Qt.Key_F12:  "F12",
    Qt.Key_F13:  "F13",
    Qt.Key_F14:  "F14",
    Qt.Key_F15:  "F15",
    Qt.Key_F16:  "F16",
    Qt.Key_F17:  "F17",
    Qt.Key_F18:  "F18",
    Qt.Key_F19:  "F19",
    Qt.Key_F20:  "F20",
    Qt.Key_F21:  "F21",
    Qt.Key_F22:  "F22",
    Qt.Key_F23:  "F23",
    Qt.Key_F24:  "F24",
}