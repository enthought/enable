""" Enable is an interactive graphical component framework built on top of Kiva.

See https://www.enthought.com/enthought/wiki/EnableProject
"""

# Major package imports
# TODO - Add basic comments for the names being imported from base and enable_traits
from .base import IDroppedOnHandler, TOP, VCENTER, BOTTOM, LEFT, HCENTER, RIGHT, \
    TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, str_to_font, \
    empty_rectangle, intersect_bounds

from .enable_traits import basic_sequence_types, sequence_types, pointer_shapes, \
     CURSOR_X, CURSOR_Y, cursor_styles, border_size_editor, font_trait, \
     bounds_trait, ComponentMinSize, ComponentMaxSize, Pointer, cursor_style_trait, \
     spacing_trait, padding_trait, margin_trait, border_size_trait, \
     TimeInterval, Stretch, NoStretch, LineStyle, LineStyleEditor

from .colors import color_table, transparent_color, ColorTrait, black_color_trait, \
                   white_color_trait, transparent_color_trait, ColorEditorFactory

from .markers import MarkerTrait, marker_trait, MarkerNameDict, marker_names, \
    SquareMarker, CircleMarker, TriangleMarker, Inverted_TriangleMarker, \
    LeftTriangleMarker, RightTriangleMarker, PentagonMarker, Hexagon1Marker,\
    Hexagon2Marker, StarMarker, CrossPlusMarker, PlusMarker, CrossMarker,\
    DiamondMarker, DotMarker, PixelMarker, CustomMarker, AbstractMarker

from .events import drag_event_trait, key_event_trait, mouse_event_trait, \
    BasicEvent, BlobEvent, BlobFrameEvent, DragEvent, KeyEvent, MouseEvent
from .interactor import Interactor
from .base_tool import BaseTool, KeySpec

from .abstract_overlay import AbstractOverlay
from .canvas import Canvas
from .component import Component
from .container import Container
from .coordinate_box import CoordinateBox
from .component_editor import ComponentEditor
from .overlay_container import OverlayContainer

try:
    import kiwisolver
except ImportError:
    pass
else:
    from .constraints_container import ConstraintsContainer
    del kiwisolver

# Breaks code that does not use numpy
from .label import Label

from .graphics_context import GraphicsContextEnable, ImageGraphicsContextEnable

# Old Enable classes and widgets
from .abstract_window import AbstractWindow

from .native_scrollbar import NativeScrollBar
from .compass import Compass
from .scrolled import Scrolled
from .slider import Slider
from .text_field_style import TextFieldStyle
from .text_field import TextField
from .text_field_grid import TextFieldGrid
from .viewport import Viewport
from .window import Window

from .primitives.api import Annotater, Box, Line, Polygon
