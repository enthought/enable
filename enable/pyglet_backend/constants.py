
from pyglet.window import BaseWindow as win
from pyglet.window import key
from pyglet.window import mouse

BUTTON_NAME_MAP = {
    mouse.LEFT : "Left",
    mouse.RIGHT : "Right",
    mouse.MIDDLE : "Middle",
}


POINTER_MAP = {
    "arrow" : win.CURSOR_DEFAULT,
    "arrow wait" : win.CURSOR_WAIT,
    "char" : win.CURSOR_TEXT,
    "cross" : win.CURSOR_CROSSHAIR,
    "hand" : win.CURSOR_HAND,
    "ibeam" : win.CURSOR_TEXT,
    "no entry" : win.CURSOR_NO,
    "question arrow" : win.CURSOR_HELP,
    # The following directions are not reversed; they have different
    # meanings between Enable and Pyglet.  Enable's 'left' and 'right'
    # designation refer to which border is being sized, whereas Pyglet's
    # means which direction the arrow points.
    "size top" : win.CURSOR_SIZE_DOWN,
    "size bottom" : win.CURSOR_SIZE_UP,
    "size left" : win.CURSOR_SIZE_RIGHT,
    "size right" : win.CURSOR_SIZE_LEFT,
    "size top right" : win.CURSOR_SIZE_DOWN,
    "size bottom left" : win.CURSOR_SIZE_DOWN,
    "size top left" : win.CURSOR_SIZE_DOWN,
    "size bottom right" : win.CURSOR_SIZE_DOWN,
    "sizing" : win.CURSOR_SIZE,
    "wait" : win.CURSOR_WAIT,
    "watch" : win.CURSOR_WAIT,
    "arrow wait" : win.CURSOR_WAIT_ARROW,

    # No good translation for these; we'll have to trap them
    # in set_pointer() or use a custom image.
    "bullseye" : win.CURSOR_DEFAULT,
    "left button" : win.CURSOR_DEFAULT,
    "middle button" : win.CURSOR_DEFAULT,
    "right button" : win.CURSOR_DEFAULT,
    "magnifier" :  win.CURSOR_DEFAULT,
    "paint brush" :  win.CURSOR_DEFAULT,
    "pencil" :  win.CURSOR_DEFAULT,
    "point left" :  win.CURSOR_DEFAULT,
    "point right" :  win.CURSOR_DEFAULT,
    "spray can" :  win.CURSOR_DEFAULT,
}

# Since Pyglet has both on_key_press and on_text events, and it's not
# entirely clear if certain keys generate both events, we maintain an
# empirical list of keys in the KEY_MAP that also generate on_text
# events.  (Generally, entries in the KEY_MAP are used to map unprintable
# or control characters, so by default we would expect them not to
# generate on_text events.)
TEXT_KEYS = (
        key.RETURN, key.ENTER, key.NUM_0, key.NUM_1, key.NUM_2,
        key.NUM_3, key.NUM_4, key.NUM_5, key.NUM_6, key.NUM_7,
        key.NUM_8, key.NUM_9, key.NUM_DECIMAL, key.NUM_ADD,
        key.NUM_MULTIPLY, key.NUM_SUBTRACT, key.NUM_DIVIDE,
        )

ASCII_CONTROL_KEYS = {
        8: "Backspace",
        9: "Tab",
        13: "Enter",
        27: "Esc",
        }

KEY_MAP = {
    key.BACKSPACE : "Backspace",
    key.TAB : "Tab",
    key.RETURN : "Enter",
    key.ESCAPE : "Esc",
    key.DELETE : "Delete",
    key.ENTER : "Enter",
    key.PAUSE : "Pause",
    key.NUMLOCK : "Num Lock",
    key.SCROLLLOCK : "Scroll Lock",
    key.MINUS : "Subtract",

    key.LEFT : "Left",
    key.RIGHT : "Right",
    key.UP : "Up",
    key.DOWN : "Down",
    key.HOME : "Home",
    key.END : "End",
    key.PAGEUP : "Page Up",
    key.PAGEDOWN : "Page Down",

    key.EXECUTE : "Execute",
    key.PRINT : "Print",
    key.SELECT : "Select",
    key.INSERT : "Insert",
    key.CANCEL : "Cancel",
    key.BREAK : "Break",
    key.HELP : "Help",

    key.NUM_0 : "Numpad 0",
    key.NUM_1 : "Numpad 1",
    key.NUM_2 : "Numpad 2",
    key.NUM_3 : "Numpad 3",
    key.NUM_4 : "Numpad 4",
    key.NUM_5 : "Numpad 5",
    key.NUM_6 : "Numpad 6",
    key.NUM_7 : "Numpad 7",
    key.NUM_8 : "Numpad 8",
    key.NUM_9 : "Numpad 9",
    key.NUM_DOWN : "Down",
    key.NUM_UP : "Up",
    key.NUM_LEFT : "Left",
    key.NUM_RIGHT : "Right",
    key.NUM_HOME : "Home",
    key.NUM_END : "End",
    key.NUM_PAGE_UP : "Page Up",
    key.NUM_PAGE_DOWN : "Page Down",
    key.NUM_ENTER : "Enter",
    key.NUM_INSERT : "Insert",
    key.NUM_DELETE : "Delete",
    key.NUM_DECIMAL : ".",
    key.NUM_ADD : "+",
    key.NUM_MULTIPLY : "*",
    key.NUM_SUBTRACT : "-",
    key.NUM_DIVIDE : "/",

    key.LSHIFT : "Shift",
    key.RSHIFT : "Shift",
    key.LCTRL : "Control",
    key.RCTRL : "Control",
    key.LMETA : "Meta",
    key.RMETA : "Meta",
    key.LALT : "Alt",
    key.RALT : "Alt",
    key.LWINDOWS : "Windows",
    key.RWINDOWS : "Windows",
    key.LCOMMAND : "Command",
    key.RCOMMAND : "Command",
    key.LOPTION : "Option",
    key.ROPTION : "Option",

    key.F1 : "F1",
    key.F2 : "F2",
    key.F3 : "F3",
    key.F4 : "F4",
    key.F5 : "F5",
    key.F6 : "F6",
    key.F7 : "F7",
    key.F8 : "F8",
    key.F9 : "F9",
    key.F10 : "F10",
    key.F11 : "F11",
    key.F12 : "F12",
    key.F13 : "F13",
    key.F14 : "F14",
    key.F15 : "F15",
    key.F16 : "F16",
    }
