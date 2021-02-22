# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# Major library imports
import os.path

# Enthought library imports
from enable.colors import ColorTrait
from traits.api import Bool, Delegate, HasTraits, PrefixList, Str, Trait
from traitsui.api import View, Group

# Local relative imports
from .base import (
    BOTTOM, EMBOSSED, ENGRAVED, HCENTER, LEFT, RIGHT, TOP, VCENTER,
    add_rectangles, transparent_color, xy_in_bounds
)
from .component import Component
from .enable_traits import (
    border_size_trait, image_trait, margin_trait, padding_trait, spacing_trait
)
from .enable_traits import position_trait, font_trait, engraving_trait
from .radio_group import RadioStyle, RadioGroup


# -----------------------------------------------------------------------------
#  Constants:
# -----------------------------------------------------------------------------

empty_text_info = (0, 0, 0, 0)
LEFT_OR_RIGHT = LEFT | RIGHT
TOP_OR_BOTTOM = TOP | BOTTOM


orientation_trait = PrefixList(["text", "component"], default_value="text")


class LabelTraits(HasTraits):

    text = Str
    font = font_trait
    text_position = position_trait("left")
    color = ColorTrait("black")
    shadow_color = ColorTrait("white")
    style = engraving_trait

    image = image_trait
    image_position = position_trait("left")
    image_orientation = orientation_trait

    spacing_height = spacing_trait
    spacing_width = spacing_trait
    padding_left = padding_trait
    padding_right = padding_trait
    padding_top = padding_trait
    padding_bottom = padding_trait
    margin_left = margin_trait
    margin_right = margin_trait
    margin_top = margin_trait
    margin_bottom = margin_trait
    border_size = border_size_trait
    border_color = ColorTrait("black")
    bg_color = ColorTrait("clear")

    enabled = Bool(True)
    selected = Bool(False)

    # -------------------------------------------------------------------------
    #  Trait view definitions:
    # -------------------------------------------------------------------------

    traits_view = View(
        Group("enabled", "selected", id="component"),
        Group(
            "text",
            " ",
            "font",
            " ",
            "color",
            " ",
            "shadow_color",
            " ",
            "style",
            id="text",
            style="custom",
        ),
        Group(
            "bg_color{Background Color}",
            "_",
            "border_color",
            "_",
            "border_size",
            id="border",
            style="custom",
        ),
        Group(
            "text_position",
            "_",
            "image_position",
            "_",
            "image_orientation",
            " ",
            "image",
            id="position",
            style="custom",
        ),
        Group(
            "spacing_height",
            "spacing_width",
            "_",
            "padding_left",
            "padding_right",
            "padding_top",
            "padding_bottom",
            "_",
            "margin_left",
            "margin_right",
            "margin_top",
            "margin_bottom",
            id="margin",
        ),
    )


default_label_traits = LabelTraits()

# -----------------------------------------------------------------------------
#  'Label' class:
# -----------------------------------------------------------------------------

LabelTraitDelegate = Delegate("delegate", redraw=True)
LayoutLabelTraitDelegate = LabelTraitDelegate(layout=True)
LabelContentDelegate = LayoutLabelTraitDelegate(content=True)


class Label(Component):

    # -------------------------------------------------------------------------
    #  Trait definitions:
    # -------------------------------------------------------------------------

    delegate = Trait(default_label_traits)
    text = LabelContentDelegate
    font = LabelContentDelegate
    text_position = LayoutLabelTraitDelegate
    color = LabelTraitDelegate
    shadow_color = LabelTraitDelegate
    style = LabelTraitDelegate

    image = LayoutLabelTraitDelegate
    image_position = LayoutLabelTraitDelegate
    image_orientation = LayoutLabelTraitDelegate

    spacing_height = LayoutLabelTraitDelegate
    spacing_width = LayoutLabelTraitDelegate
    padding_left = LayoutLabelTraitDelegate
    padding_right = LayoutLabelTraitDelegate
    padding_top = LayoutLabelTraitDelegate
    padding_bottom = LayoutLabelTraitDelegate
    margin_left = LayoutLabelTraitDelegate
    margin_right = LayoutLabelTraitDelegate
    margin_top = LayoutLabelTraitDelegate
    margin_bottom = LayoutLabelTraitDelegate
    border_size = LayoutLabelTraitDelegate
    border_color = LabelTraitDelegate
    bg_color = LabelTraitDelegate

    enabled = LabelTraitDelegate
    selected = LabelTraitDelegate

    # -------------------------------------------------------------------------
    #  Trait view definitions:
    # -------------------------------------------------------------------------

    traits_view = View(
        Group("<component>", "enabled", "selected", id="component"),
        Group("<links>", "delegate", id="links"),
        Group(
            "text",
            " ",
            "font",
            " ",
            "color",
            " ",
            "shadow_color",
            " ",
            "style",
            id="text",
            style="custom",
        ),
        Group(
            "bg_color{Background Color}",
            "_",
            "border_color",
            "_",
            "border_size",
            id="border",
            style="custom",
        ),
        Group(
            "text_position",
            "_",
            "image_position",
            "_",
            "image_orientation",
            " ",
            "image",
            id="position",
            style="custom",
        ),
        Group(
            "spacing_height",
            "spacing_width",
            "_",
            "padding_left",
            "padding_right",
            "padding_top",
            "padding_bottom",
            "_",
            "margin_left",
            "margin_right",
            "margin_top",
            "margin_bottom",
            id="margin",
        ),
    )

    colorchip_map = {
        "fg_color": "color",
        "bg_color": "bg_color",
        "shadow_color": "shadow_color",
        "alt_color": "border_color",
    }

    # -------------------------------------------------------------------------
    #  Initialize the object:
    # -------------------------------------------------------------------------

    def __init__(self, text="", **traits):
        self.text = text
        Component.__init__(self, **traits)

    # -------------------------------------------------------------------------
    #  Handle any trait being modified:
    # -------------------------------------------------------------------------

    def _anytrait_changed(self, name, old, new):
        trait = self.trait(name)
        if trait.content:
            self.update_text()
        if trait.redraw:
            if trait.layout:
                self.layout()
            self.redraw()

    # -------------------------------------------------------------------------
    #  Return the components that contain a specified (x,y) point:
    # -------------------------------------------------------------------------

    def _components_at(self, x, y):
        if self._in_margins(x, y):
            return [self]
        return []

    # -------------------------------------------------------------------------
    #  Return whether not a specified point is inside the component margins:
    # -------------------------------------------------------------------------

    def _in_margins(self, x, y):
        ml = self.margin_left
        mb = self.margin_bottom
        return xy_in_bounds(
            x,
            y,
            add_rectangles(
                self.bounds,
                (ml, mb, -(self.margin_right + ml), -(self.margin_top + mb)),
            ),
        )

    # -------------------------------------------------------------------------
    #  Update any information related to the text content of the control:
    # -------------------------------------------------------------------------

    def update_text(self):
        text = self.text
        if text == "":
            self._text = []
            self._tdx = []
            self._max_tdx = self._tdy = 0
        else:
            self._text = _text = text.split("\n")
            gc = self.gc_temp()
            gc.set_font(self.font)
            max_tdx = 0
            self._tdx = _tdx = [0] * len(_text)
            for i, text in enumerate(_text):
                tdx, tdy, descent, leading = gc.get_full_text_extent(text)
                tdy += descent + 5
                max_tdx = max(max_tdx, tdx)
                _tdx[i] = tdx
            self._max_tdx = max_tdx
            self._tdy = tdy

    # -------------------------------------------------------------------------
    #  Layout and compute the minimum size of the control:
    # -------------------------------------------------------------------------

    def layout(self):
        sdx = self.spacing_width
        sdy = self.spacing_height
        n = len(self._text)
        if n == 0:
            tdx = tdy = sdx = sdy = 0
        else:
            tdx = self._max_tdx
            tdy = self._tdy * n
        image = self._image
        if image is not None:
            idx = image.width()
            idy = image.height()
        else:
            idx = idy = sdx = sdy = 0
        image_position = self.image_position_
        if image_position & LEFT_OR_RIGHT:
            itdx = tdx + sdx + idx
            if image_position & LEFT:
                ix = 0
                tx = idx + sdx
            else:
                tx = 0
                ix = tdx + sdx
        else:
            itdx = max(tdx, idx)
            ix = (itdx - idx) / 2.0
            tx = (itdx - tdx) / 2.0
        if image_position & TOP_OR_BOTTOM:
            itdy = tdy + sdy + idy
            if image_position & TOP:
                iy = tdy + sdy
                ty = 0
            else:
                iy = 0
                ty = idy + sdy
        else:
            itdy = max(tdy, idy)
            iy = (itdy - idy) / 2.0
            ty = (itdy - tdy) / 2.0
        bs = 2 * self.border_size
        self.min_width = itdx + (
            self.margin_left
            + self.margin_right
            + self.padding_left
            + self.padding_right
            + bs
        )
        self.min_height = itdy + (
            self.margin_top
            + self.margin_bottom
            + self.padding_top
            + self.padding_bottom
            + bs
        )
        self._info = (ix, iy, idx, idy, tx, ty, tdx, self._tdy, itdx, itdy)

    # -------------------------------------------------------------------------
    #  Draw the contents of the control:
    # -------------------------------------------------------------------------

    def _draw(self, gc, view_bounds, mode):

        # Set up all the control variables for quick access:
        ml = self.margin_left
        mr = self.margin_right
        mt = self.margin_top
        mb = self.margin_bottom
        pl = self.padding_left
        pr = self.padding_right
        pt = self.padding_top
        pb = self.padding_bottom
        bs = self.border_size
        bsd = bs + bs
        bsh = bs / 2.0
        x, y, dx, dy = self.bounds

        ix, iy, idx, idy, tx, ty, tdx, tdy, itdx, itdy = self._info

        # Fill the background region (if required);
        bg_color = self.bg_color_
        if bg_color is not transparent_color:
            with gc:
                gc.set_fill_color(bg_color)
                gc.begin_path()
                gc.rect(
                    x + ml + bs,
                    y + mb + bs,
                    dx - ml - mr - bsd,
                    dy - mb - mt - bsd,
                )
                gc.fill_path()

        # Draw the border (if required):
        if bs > 0:
            border_color = self.border_color_
            if border_color is not transparent_color:
                with gc:
                    gc.set_stroke_color(border_color)
                    gc.set_line_width(bs)
                    gc.begin_path()
                    gc.rect(
                        x + ml + bsh,
                        y + mb + bsh,
                        dx - ml - mr - bs,
                        dy - mb - mt - bs,
                    )
                    gc.stroke_path()

        # Calculate the origin of the image/text box:
        text_position = self.text_position_

        if self.image_orientation == "text":
            # Handle the 'image relative to text' case:
            if text_position & RIGHT:
                itx = x + dx - mr - bs - pr - itdx
            else:
                itx = x + ml + bs + pl
                if text_position & HCENTER:
                    itx += (dx - ml - mr - bsd - pl - pr - itdx) / 2.0
            if text_position & TOP:
                ity = y + dy - mt - bs - pt - itdy
            else:
                ity = y + mb + bs + pb
                if text_position & VCENTER:
                    ity += (dy - mb - mt - bsd - pb - pt - itdy) / 2.0
        else:
            # Handle the 'image relative to component' case:
            itx = ity = 0.0
            if text_position & RIGHT:
                tx = x + dx - mr - bs - pr - tdx
            else:
                tx = x + ml + bs + pl
                if text_position & HCENTER:
                    tx += (dx - ml - mr - bsd - pl - pr - tdx) / 2.0
            if text_position & TOP:
                ty = y + dy - mt - bs - pt - tdy
            else:
                ty = y + mb + bs + pb
                if text_position & VCENTER:
                    ty += (dy - mb - mt - bsd - pb - pt - tdy) / 2.0

            image_position = self.image_position_
            if image_position & RIGHT:
                ix = x + dx - mr - bs - pr - idx
            else:
                ix = x + ml + bs + pl
                if image_position & HCENTER:
                    ix += (dx - ml - mr - bsd - pl - pr - idx) / 2.0
            if image_position & TOP:
                iy = y + dy - mt - bs - pt - idy
            else:
                iy = y + mb + bs + pb
                if image_position & VCENTER:
                    iy += (dy - mb - mt - bsd - pb - pt - idy) / 2.0

        with gc:
            # Draw the image (if required):
            image = self._image
            if image is not None:
                gc.draw_image(image, (itx + ix, ity + iy, idx, idy))

            # Draw the text (if required):
            gc.set_font(self.font)
            _text = self._text
            _tdx = self._tdx
            tx += itx
            ty += ity + tdy * len(_text)
            style = self.style_
            shadow_color = self.shadow_color_
            text_color = self.color_
            for i, text in enumerate(_text):
                ty -= tdy
                _tx = tx
                if text_position & RIGHT:
                    _tx += tdx - _tdx[i]
                elif text_position & HCENTER:
                    _tx += (tdx - _tdx[i]) / 2.0
                # Draw the 'shadow' text, if requested:
                if (style != 0) and (shadow_color is not transparent_color):
                    if style == EMBOSSED:
                        gc.set_fill_color(shadow_color)
                        gc.set_text_position(_tx - 1.0, ty + 1.0)
                    elif style == ENGRAVED:
                        gc.set_fill_color(shadow_color)
                        gc.set_text_position(_tx + 1.0, ty - 1.0)
                    else:
                        gc.set_fill_color(shadow_color)
                        gc.set_text_position(_tx + 2.0, ty - 2.0)
                    gc.show_text(text)

                # Draw the normal text:
                gc.set_fill_color(text_color)
                gc.set_text_position(_tx, ty)
                gc.show_text(text)

    # -- Pickling Protocol ----------------------------------------------------

    def __getstate__(self):
        dict = self.__dict__.copy()
        try:
            del dict["_image"]
        except Exception:
            pass
        return dict

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.image = self.image


# -----------------------------------------------------------------------------
#  'CheckBox' class:
# -----------------------------------------------------------------------------


class CheckBox(Label):

    # -------------------------------------------------------------------------
    #  Trait definitions:
    # -------------------------------------------------------------------------

    image_base = Str("=checkbox")

    # -------------------------------------------------------------------------
    #  Trait editor definition:
    # -------------------------------------------------------------------------

    position = Group("<position>", "image_base")

    # -------------------------------------------------------------------------
    #  Initialize the object:
    # -------------------------------------------------------------------------

    def __init__(self, text="", **traits):
        Label.__init__(self, text, **traits)
        self._select_image()

    # -------------------------------------------------------------------------
    #  Select the correct image to display:
    # -------------------------------------------------------------------------

    def _select_image(self, *suffixes):
        if len(suffixes) == 0:
            suffixes = [self._suffix()]
        base, ext = os.path.splitext(self.image_base)
        for suffix in suffixes:
            image = "%s%s%s" % (base, suffix, ext)
            if self.image_for(image) is not None:
                self.image = image
                break

    # -------------------------------------------------------------------------
    #  Select the image suffix based on the current selection state:
    # -------------------------------------------------------------------------

    def _suffix(self):
        return ["", "_on"][self.selected]

    # -------------------------------------------------------------------------
    #  Set the selection state of the component:
    # -------------------------------------------------------------------------

    def _select(self):
        self.selected = not self.selected

    # -------------------------------------------------------------------------
    #  Handle the 'selected' status of the checkbox being changed:
    # -------------------------------------------------------------------------

    def _selected_changed(self):
        base = self._suffix()
        self._select_image(base + ["", "_over"][self._over is True], base)

    # -------------------------------------------------------------------------
    #  Handle mouse events:
    # -------------------------------------------------------------------------

    def _left_down_changed(self, event):
        event.handled = True
        if self._in_margins(event.x, event.y):
            event.window.mouse_owner = self
            base = self._suffix()
            self._select_image(base + "_down", base)
            self._down = True

    def _left_dclick_changed(self, event):
        self._left_down_changed(event)

    def _left_up_changed(self, event):
        event.handled = True
        event.window.mouse_owner = self._down = None
        if self._in_margins(event.x, event.y):
            self._select()

    def _mouse_move_changed(self, event):
        event.handled = True
        self._over = self._in_margins(event.x, event.y)
        if self._over:
            event.window.mouse_owner = self
            base = self._suffix()
            self._select_image(
                base + ["_over", "_down"][self._down is not None], base
            )
        else:
            if self._down is None:
                event.window.mouse_owner = None
            self._select_image()


# -----------------------------------------------------------------------------
#  'RadioButton' class:
# -----------------------------------------------------------------------------


class Radio(CheckBox, RadioStyle):

    # -------------------------------------------------------------------------
    #  Trait definitions:
    # -------------------------------------------------------------------------

    image_base = Str("=radio")

    # -------------------------------------------------------------------------
    #  Set the selection state of the component:
    # -------------------------------------------------------------------------

    def _select(self):
        self.selected = True

    # -------------------------------------------------------------------------
    #  Handle the container the component belongs to being changed:
    # -------------------------------------------------------------------------

    def _container_changed(self, old, new):
        CheckBox._container_changed(self)
        if self.radio_group is old.radio_group:
            self.radio_group = None
        if self.radio_group is None:
            if new.radio_group is None:
                new.radio_group = RadioGroup()
            new.radio_group.add(self)

    # -------------------------------------------------------------------------
    #  Handle the 'selected' status of the checkbox being changed:
    # -------------------------------------------------------------------------

    def _selected_changed(self):
        CheckBox._selected_changed(self)
        if self.selected:
            self.radio_group.selection = self
