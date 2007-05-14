""" Enable is an interactive graphical component framework built on top of Kiva.

See https://www.enthought.com/enthought/wiki/EnableProject
"""

# Major package imports
# TODO - Add basic comments for the names being imported from base and enable_traits
from base import IDroppedOnHandler, TOP, VCENTER, BOTTOM, LEFT, HCENTER, RIGHT, \
    TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, str_to_font, filled_rectangle, \
    empty_rectangle, intersect_bounds

from enable_traits import basic_sequence_types, sequence_types, pointer_shapes, \
     CURSOR_X, CURSOR_Y, cursor_styles, TraitImage, border_size_editor, font_trait, \
     bounds_trait, ComponentMinSize, ComponentMaxSize, Pointer, cursor_style_trait, \
     engraving_trait, spacing_trait, padding_trait, margin_trait, border_size_trait, \
     image_trait, string_image_trait, TimeInterval, Stretch, NoStretch, LineStyle, \
     LineStyleEditor

from colors import color_table, transparent_color, ColorTrait, black_color_trait, \
                   white_color_trait, transparent_color_trait, ColorEditorFactory

from events import drag_event_trait, key_event_trait, mouse_event_trait, \
    DragEvent, KeyEvent, MouseEvent
from interactor import Interactor

from component import Component
#from component_render_category import ComponentRenderCategory
#from component_layout_category import ComponentLayoutCategory
from container import Container
from coordinate_box import CoordinateBox
#from drag import DragHandler
#from drag_resize import DragResizeHandler

# Breaks code that does not use numpy
from text_grid import TextGrid
from viewport import Viewport

from graphics_context import GraphicsContextEnable

# Old Enable classes and widgets

from abstract_window import AbstractWindow
#from controls import LabelTraits, Label, CheckBox, Radio
#This is not the way we want it, but we need to come up 
#with a good system of handing imports when we have multiple
#underlying widget sets.
from wx_backend.scrollbar import NativeScrollBar
from scrolled import Scrolled
#from key_bindings import KeyBinding, KeyBindings
#from color_picker import ColorPicker

# subpackage imports
from image.api import Image, DraggableImage, Inspector, ColorChip
#from image_frame import ImageFrame, ResizeFrame, TitleFrame, WindowFrame, \
#                        ComponentFactory, Button, CheckBoxButton, RadioButton
#from image_title import ImageTitle
#from drawing_canvas import GriddedCanvas, GuideLine, SelectionFrame
from primitives.api import Annotater, Box, Line, Polygon
