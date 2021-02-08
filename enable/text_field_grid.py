# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# Enthought library imports
from enable.api import Container, TextField
from traits.api import Float, Int, List

# Local imports
from .enable_traits import LineStyle
from .colors import black_color_trait


class TextFieldGrid(Container):
    """ A 2D grid of TextFields.
    """

    #########################################################################
    # TextFieldGrid traits
    #########################################################################

    # The number of columns
    columns = Int(0)

    # The number of rows
    rows = Int(0)

    # The Cells in the TextGrid - represented as a list of lists
    cells = List(List)

    # The initial width of the boxes
    cell_width = Float(200.0)

    # The initial height of the boxes
    cell_height = Float(20.0)

    # The padding between the boxes
    cell_padding = Float(1.0)

    # The thickness of the border between cells
    cell_border_width = Int(1)

    # The color of the border between cells
    cell_border_color = black_color_trait

    # The dash style of the border between cells
    cell_border_style = LineStyle("solid")

    # A list of tuples of the (i,j) of selected cells
    selected_cells = List

    # The total width of the textfields after layout
    total_width = Float

    # The total height of the textfields after layout
    total_height = Float

    #########################################################################
    # object interface
    #########################################################################

    def __init__(self, columns, rows, **traits):
        """ Create a list of lists of EnableTextFields.  These will be
            the elements in our TextGrid.
        """
        self.rows = rows
        self.columns = columns
        self.selected_box = []
        super(TextFieldGrid, self).__init__(**traits)

    def set_cell(self, row, column, text):
        if row < self.rows and column < self.columns:
            self.cells[row][column].text = text

    #########################################################################
    # AbstractComponent interface
    #########################################################################

    def _dispatch_draw(self, layer, gc, view_bounds, mode):
        self._position_cells()
        Container._dispatch_draw(self, layer, gc, view_bounds, mode)
        self._draw_grid(gc, view_bounds, mode)

    # Private drawing methods ############################################

    def _draw_grid(self, gc, view_bounds, mode):
        gc.set_stroke_color(self.cell_border_color_)
        gc.set_line_dash(self.cell_border_style_)
        gc.set_antialias(0)

        y = self.y
        for row in range(self.rows + 1):
            gc.move_to(self.x, y)
            gc.line_to(self.x + self.width, y)
            gc.stroke_path()
            y = y + self.cell_padding + self.cell_height

        x = self.x
        for cell in range(self.columns + 1):
            gc.move_to(x, self.y)
            gc.line_to(x, self.y + self.height)
            gc.stroke_path()
            x = x + self.cell_padding + self.cell_width

    def _position_cells(self):
        y = 0
        for row in self.cells:
            x = 0
            for cell in row:
                cell.position = [x, y]
                x = x + self.cell_padding + self.cell_width
            y = y + self.cell_padding + self.cell_height
        self.total_width = x
        self.total_height = y

    def _add_row(self, index):
        row = []
        for i in range(self.columns):
            tfield = TextField(
                position=[0, 0],
                width=self.cell_width,
                height=self.cell_height,
                multiline=False,
            )
            self.add(tfield)
            row.append(tfield)
        self.cells.insert(index, row)
        self.bounds[1] = self.bounds[1] + self.cell_padding + self.cell_height

    def _add_column(self, index):
        for row in self.cells:
            tfield = TextField(
                position=[0, 0],
                width=self.cell_width,
                height=self.cell_height,
                multiline=False,
            )
            self.add(tfield)
            row.insert(index, tfield)
        self.bounds[0] = self.bounds[0] + self.cell_padding + self.cell_width

    def _remove_row(self, index):
        removed = self.cells[index]
        self.remove(removed)
        self.cells.remove(removed)
        self.bounds[1] = self.bounds[1] - self.cell_padding - self.cell_height

    def _remove_column(self, index):
        for row in self.cells:
            removed = row[index]
            self.remove(removed)
            row.remove(removed)
        self.bounds[0] = self.bounds[0] - self.cell_padding - self.cell_width

    #########################################################################
    # TextFieldGrid interface
    #########################################################################

    def _rows_changed(self, old, new):
        if new > old:
            for i in range(old, new):
                self._add_row(i)
        else:
            for i in range(new, old):
                self._remove_row(i)
        self.request_redraw()

    def _columns_changed(self, old, new):
        if new > old:
            for i in range(old, new):
                self._add_column(i)
        else:
            for i in range(new, old):
                self._remove_column(i)
        self.request_redraw()

    def _cells_changed(self, new):
        self.request_redraw()


# Test
if __name__ == "__main__":
    from enable.wx_backend.api import Window
    from enable.api import Container
    from enable.example_support import DemoFrame, demo_main

    class MyFrame(DemoFrame):
        def _create_window(self):
            box1 = TextFieldGrid(4, 2, position=[50, 300])

            box1.set_cell(1, 1, "apple")
            box1.set_cell(0, 3, "pear")

            container = Container(bounds=[800, 600], use_backbuffer=False)
            container.add(box1)
            return Window(self, -1, size=[800, 600], component=container)

    demo_main(MyFrame)
