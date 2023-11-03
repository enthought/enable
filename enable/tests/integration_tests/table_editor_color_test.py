# (C) Copyright 2004-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""Integration test for Enable ColorTrait and TraitsUI ColorColumn

This test makes sure that the Enable ColorTrait works in the context of a
ColorColumn of a TraitsUI TableEditor
"""

from traits.api import HasTraits, List
from traitsui.api import View, Item, TableEditor
from traitsui.color_column import ColorColumn

from enable.api import ColorTrait


class Thingy(HasTraits):
    color = ColorTrait('black')


colors = [
    Thingy(color='red'),
    Thingy(color='orange'),
    Thingy(color='yellow'),
    Thingy(color='green'),
    Thingy(color='blue'),
    Thingy(color='indigo'),
    Thingy(color='violet'),
    Thingy(color='black'),
    Thingy(color='white'),
]


class TableTest(HasTraits):

    colors = List(Thingy)

    table_editor = TableEditor(
        columns=[
            ColorColumn(name='color'),
        ],
        editable=True,
        deletable=True,
        sortable=True,  #
        sort_model=True,
        show_lines=True,  #
        orientation='vertical',
        show_column_labels=True,  #
        row_factory=Thingy,
    )

    traits_view = View(
        [Item('colors', id='colors', editor=table_editor), '|[]<>'],
        title='Table Editor Test',
        id='traitsui.tests.table_editor_color_test',
        dock='horizontal',
        width=0.4,
        height=0.3,
        resizable=True,
        kind='live',
    )


if __name__ == '__main__':
    tt = TableTest(colors=colors)
    tt.configure_traits()