# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from numpy import array

from kiva.api import STROKE
from kiva.image import CompiledPath, GraphicsContext

cross = CompiledPath()
cross.scale_ctm(10.0, 10.0)
lines = array(
    [
        (0, 1), (0, 2), (1, 2), (1, 3),
        (2, 3), (2, 2), (3, 2), (3, 1),
        (2, 1), (2, 0), (1, 0), (1, 1),
        (0, 1),
    ]
)
cross.lines(lines)

gc = GraphicsContext((400, 400))
gc.set_stroke_color((1, 0, 0, 1))
gc.draw_path_at_points(
    array([(50, 50), (200, 50), (50, 200), (200, 200)]), cross, STROKE
)
gc.save("compiled_path.png")
