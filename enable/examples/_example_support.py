# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""
Support class that wraps up the boilerplate toolkit calls that virtually all
demo programs have to use.
"""

from traits.api import HasTraits, Instance
from traits.etsconfig.api import ETSConfig
from traitsui.api import Item, View

from enable.api import Component, ComponentEditor


class DemoFrame(HasTraits):

    component = Instance(Component)

    traits_view = View(
        Item("component", editor=ComponentEditor(), show_label=False),
        resizable=True,
    )

    def _component_default(self):
        return self._create_component()

    def _create_component(self):
        """ Create and return a component which is typically a
        container with nested components """
        raise NotImplementedError


def demo_main(demo_class, size=(640, 480), title="Enable Example"):
    demo_class().configure_traits()
