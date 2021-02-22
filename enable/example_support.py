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

# FIXME - it should be enough to do the following import, but because of the
# PyQt/traits problem (see below) we can't because it would drag in traits too
# early.  Until it is fixed we just assume wx if we can import it.
# Force the selection of a valid toolkit.
# import enable.toolkit
if not ETSConfig.toolkit:
    for toolkit, toolkit_module in (("wx", "wx"), ("qt4", "PyQt4")):
        try:
            exec("import " + toolkit_module)
            ETSConfig.toolkit = toolkit
            break
        except ImportError:
            pass
    else:
        raise RuntimeError("Can't load wx or qt4 backend for Chaco.")


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
