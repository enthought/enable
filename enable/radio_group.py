# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# -----------------------------------------------------------------------------
#  Imports:
# -----------------------------------------------------------------------------

from traits.api import HasPrivateTraits, Trait

# -----------------------------------------------------------------------------
#  Trait definitions:
# -----------------------------------------------------------------------------

# ARadioGroup = Instance( 'RadioGroup' )

# -----------------------------------------------------------------------------
#  'RadioStyle' class:
# -----------------------------------------------------------------------------


class RadioStyle(HasPrivateTraits):

    # -------------------------------------------------------------------------
    #  Trait definitions:
    # -------------------------------------------------------------------------

    #    radio_group = ARadioGroup

    # -------------------------------------------------------------------------
    #  Handle the group the radio style component belongs to being changed:
    # -------------------------------------------------------------------------

    def _group_changed(self, old, new):
        if old is not None:
            old.remove(self)
        if new is not None:
            new.add(self)


# -----------------------------------------------------------------------------
#  'RadioGroup' class:
# -----------------------------------------------------------------------------


class RadioGroup(HasPrivateTraits):

    # -------------------------------------------------------------------------
    #  Trait definitions:
    # -------------------------------------------------------------------------

    # selection = Instance( RadioStyle )
    selection = Trait(None, RadioStyle)

    # -------------------------------------------------------------------------
    #  Handle elements being added to the group:
    # -------------------------------------------------------------------------

    def add(self, *components):
        for component in components:
            component.radio_group = self
            if component.selected and (self.selection is not component):
                if self.selection is not None:
                    component.selected = False
                else:
                    self.selection = component

    # -------------------------------------------------------------------------
    #  Handle components being removed from the group:
    # -------------------------------------------------------------------------

    def remove(self, *components):
        for component in components:
            if component is self.selection:
                self.selection is None
                break

    # -------------------------------------------------------------------------
    #  Handle the selection being changed:
    # -------------------------------------------------------------------------

    def _selection_changed(self, old, new):
        if old is not None:
            old.selected = False


radio_group_trait = Trait(None, RadioGroup)

RadioStyle.add_class_trait("radio_group", radio_group_trait)
