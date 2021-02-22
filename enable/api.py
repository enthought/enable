# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Enable is an interactive graphical component framework built on top of
Kiva.

See https://www.enthought.com/enthought/wiki/EnableProject

Enable Base
===========

- :class:`~.IDroppedOnHandler`
- :func:`~.str_to_font`
- :func:`~.intersect_bounds`

Constants
---------
- :attr:`~.TOP`
- :attr:`~.VCENTER`
- :attr:`~.BOTTOM`
- :attr:`~.LEFT`
- :attr:`~.HCENTER`
- :attr:`~.RIGHT`
- :attr:`~.TOP_LEFT`
- :attr:`~.TOP_RIGHT`
- :attr:`~.BOTTOM_LEFT`
- :attr:`~.BOTTOM_RIGHT`
- :attr:`~.empty_rectangle`

Enable Trait Types
==================

- :attr:`~.border_size_editor`
- :attr:`~.font_trait`
- :attr:`~.bounds_trait`
- :attr:`~.ComponentMinSize`
- :attr:`~.ComponentMaxSize`
- :attr:`~.Pointer`
- :attr:`~.cursor_style_trait`
- :attr:`~.spacing_trait`
- :attr:`~.padding_trait`
- :attr:`~.margin_trait`
- :attr:`~.border_size_trait`
- :attr:`~.TimeInterval`
- :attr:`~.Stretch`
- :attr:`~.NoStretch`
- :attr:`~.LineStyle`
- :attr:`~.LineStyleEditor`

Constants
---------
- :attr:`~.basic_sequence_types`
- :attr:`~.sequence_types`
- :attr:`~.pointer_shapes`
- :attr:`~.CURSOR_X`
- :attr:`~.CURSOR_Y`
- :attr:`~.cursor_styles`

Colors
======

- :attr:`~.color_table`
- :attr:`~.transparent_color`
- :class:`~.ColorEditorFactory`

Color Trait Types
-----------------

- :class:`~.ColorTrait`
- :attr:`~.black_color_trait`
- :attr:`~.white_color_trait`
- :attr:`~.transparent_color_trait`

Markers
=======

- :class:`~.SquareMarker`
- :class:`~.CircleMarker`
- :class:`~.TriangleMarker`
- :class:`~.Inverted_TriangleMarker`
- :class:`~.LeftTriangleMarker`
- :class:`~.RightTriangleMarker`
- :class:`~.PentagonMarker`
- :class:`~.Hexagon1Marker`
- :class:`~.Hexagon2Marker`
- :class:`~.StarMarker`
- :class:`~.CrossPlusMarker`
- :class:`~.PlusMarker`
- :class:`~.CrossMarker`
- :class:`~.DiamondMarker`
- :class:`~.DotMarker`
- :class:`~.PixelMarker`
- :class:`~.CustomMarker`
- :class:`~.AbstractMarker`

Marker Trait Types
------------------
- :class:`~.MarkerTrait`
- :attr:`~.marker_trait`

Marker Constants
----------------
- :attr:`~.MarkerNameDict`
- :attr:`~.marker_names`

Events
======

- :class:`~.BasicEvent`
- :class:`~.BlobEvent`
- :class:`~.BlobFrameEvent`
- :class:`~.DragEvent`
- :class:`~.KeyEvent`
- :class:`~.MouseEvent`

Event Trait Types
-----------------

- :attr:`~.drag_event_trait`
- :attr:`~.key_event_trait`
- :attr:`~.mouse_event_trait`

Enable Components
=================

- :class:`~.Interactor`
- :class:`~.BaseTool`
- :class:`~.KeySpec`
- :class:`~.AbstractOverlay`
- :class:`~.Canvas`
- :class:`~.Component`
- :class:`~.Container`
- :class:`~.CoordinateBox`
- :class:`~.ComponentEditor`
- :class:`~.OverlayContainer`
- :class:`~.ConstraintsContainer`
- :class:`~.Label`
- :class:`~.GraphicsContextEnable`
- :class:`~.ImageGraphicsContextEnable`

Enable Widgets
==============

- :class:`~.AbstractWindow`
- :class:`~.Viewport`
- :class:`~.Window`

Drawing Primitives
==================

- :class:`~.Annotater`
- :class:`~.Box`
- :class:`~.Line`
- :class:`~.Polygon`

"""

# Major package imports
# TODO - Add basic comments for the names being imported from base and
# enable_traits
from .base import (
    IDroppedOnHandler,
    TOP,
    VCENTER,
    BOTTOM,
    LEFT,
    HCENTER,
    RIGHT,
    TOP_LEFT,
    TOP_RIGHT,
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
    str_to_font,
    empty_rectangle,
    intersect_bounds,
)

from .enable_traits import (
    basic_sequence_types,
    sequence_types,
    pointer_shapes,
    CURSOR_X,
    CURSOR_Y,
    cursor_styles,
    border_size_editor,
    font_trait,
    bounds_trait,
    ComponentMinSize,
    ComponentMaxSize,
    Pointer,
    cursor_style_trait,
    spacing_trait,
    padding_trait,
    margin_trait,
    border_size_trait,
    TimeInterval,
    Stretch,
    NoStretch,
    LineStyle,
    LineStyleEditor,
)

from .colors import (
    color_table,
    transparent_color,
    ColorTrait,
    black_color_trait,
    white_color_trait,
    transparent_color_trait,
    ColorEditorFactory,
)

from .markers import (
    MarkerTrait,
    marker_trait,
    MarkerNameDict,
    marker_names,
    SquareMarker,
    CircleMarker,
    TriangleMarker,
    Inverted_TriangleMarker,
    LeftTriangleMarker,
    RightTriangleMarker,
    PentagonMarker,
    Hexagon1Marker,
    Hexagon2Marker,
    StarMarker,
    CrossPlusMarker,
    PlusMarker,
    CrossMarker,
    DiamondMarker,
    DotMarker,
    PixelMarker,
    CustomMarker,
    AbstractMarker,
)

from .events import (
    drag_event_trait,
    key_event_trait,
    mouse_event_trait,
    BasicEvent,
    BlobEvent,
    BlobFrameEvent,
    DragEvent,
    KeyEvent,
    MouseEvent,
)
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
