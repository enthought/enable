"""Define a simple filled box component"""

# Enthought library imports
from enthought.traits.ui.api import Group, View, Include

# Parent package imports
from enthought.enable2.api import border_size_trait, Component, transparent_color
from enthought.enable2.colors import ColorTrait


class Box(Component):

    color        = ColorTrait("white")
    border_color = ColorTrait("black")
    border_size  = border_size_trait

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        "Draw the box background in a specified graphics context"

        gc.save_state()

        # Set up all the control variables for quick access:
        bs  = self.border_size
        bsd = bs + bs
        bsh = bs / 2.0
        x, y = self.position
        dx, dy = self.bounds

        # Fill the background region (if required);
        color = self.color_
        if color is not transparent_color:
            gc.set_fill_color(color)
            gc.begin_path()
            gc.rect(x + bs, y + bs, dx - bsd, dy - bsd)
            gc.fill_path()

        # Draw the border (if required):
        if bs > 0:
            border_color = self.border_color_
            if border_color is not transparent_color:
                gc.set_stroke_color(border_color)
                gc.set_line_width(bs)
                gc.begin_path()
                gc.rect(x + bsh, y + bsh, dx - bs, dy - bs)
                gc.stroke_path()

        gc.restore_state()
        return

# EOF
