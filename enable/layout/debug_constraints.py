from collections import defaultdict

from enable.abstract_overlay import AbstractOverlay
from enable.colors import ColorTrait
from enable.enable_traits import LineStyle
from traits.api import Any, Bool, Float, HasTraits, Instance, List, Property


class Coords(HasTraits):
    """ Simple holder of box-related data.

    """
    top = Float()
    bottom = Float()
    left = Float()
    right = Float()
    width = Float()
    height = Float()

    v_center = Property()
    _v_center = Any()
    def _get_v_center(self):
        if self._v_center is None:
            return self.bottom + 0.5 * self.height
        else:
            return self._v_center
    def _set_v_center(self, value):
        self._v_center = value

    h_center = Property()
    _h_center = Any()
    def _get_h_center(self):
        if self._h_center is None:
            return self.left + 0.5 * self.width
        else:
            return self._h_center
    def _set_h_center(self, value):
        self._h_center = value


class DebugConstraintsOverlay(AbstractOverlay):
    """ Highlight the selected constraints on the outline view.

    """

    selected_constraints = List()

    # Map from box name to Coords.
    boxes = Any()

    # Style options for the lines.
    term_color = ColorTrait('orange')
    term_line_style = LineStyle('solid')

    def update_from_constraints(self, layout_mgr):
        """ Update the constraints boxes.

        """
        self.boxes = defaultdict(Coords)
        if layout_mgr is not None and layout_mgr._constraints:
            for constraint in layout_mgr._constraints:
                for expr in (constraint.lhs, constraint.rhs):
                    for term in expr.terms:
                        name, attr = self.split_var_name(term.var.name)
                        setattr(self.boxes[name], attr, term.var.value)
        self.request_redraw()

    def split_var_name(self, var_name):
        class_name, hexid, attr = var_name.rsplit('|', 2)
        name = '{}|{}'.format(class_name, hexid)
        return name, attr

    def overlay(self, other_component, gc, view_bounds=None, mode="normal"):
        """ Draws this component overlaid on another component.

        """
        if len(self.selected_constraints) == 0:
            return
        origin = other_component.position
        with gc:
            gc.translate_ctm(*origin)
            gc.set_stroke_color(self.term_color_)
            gc.set_line_dash(self.term_line_style_)
            gc.set_line_width(3)
            term_attrs = set()
            for constraint in self.selected_constraints:
                for expr in (constraint.lhs, constraint.rhs):
                    for term in expr.terms:
                        term_attrs.add(self.split_var_name(term.var.name))
            for name, attr in sorted(term_attrs):
                box = self.boxes[name]
                if attr == 'top':
                    self.hline(gc, box.left, box.top, box.width)
                elif attr == 'bottom':
                    self.hline(gc, box.left, box.bottom, box.width)
                elif attr == 'left':
                    self.vline(gc, box.left, box.bottom, box.height)
                elif attr == 'right':
                    self.vline(gc, box.right, box.bottom, box.height)
                elif attr == 'width':
                    self.hline(gc, box.left, box.v_center, box.width)
                elif attr == 'height':
                    self.vline(gc, box.h_center, box.bottom, box.height)
                gc.stroke_path()

    def vline(self, gc, x, y0, length):
        """ Draw a vertical line.

        """
        gc.move_to(x, y0)
        gc.line_to(x, y0+length)

    def hline(self, gc, x0, y, length):
        """ Draw a horizontal line.

        """
        gc.move_to(x0, y)
        gc.line_to(x0+length, y)

