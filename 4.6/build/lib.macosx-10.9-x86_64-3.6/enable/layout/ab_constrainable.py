# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from abc import ABCMeta


class ABConstrainable(object, metaclass=ABCMeta):
    """ An abstract base class for objects that can be laid out using
    layout helpers.

    Minimally, instances need to have `top`, `bottom`, `left`, `right`,
    `layout_width`, `layout_height`, `v_center` and `h_center` attributes
    which are `LinearSymbolic` instances.

    """
