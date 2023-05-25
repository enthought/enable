# (C) Copyright 2005-2023 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from kiva import agg

gc = agg.GraphicsContextArray((500, 500))
gc.rect(100, 100, 300, 300)
gc.draw_path()
gc.save("rect.bmp")

# directly manipulate the underlying
# Numeric array.
gc.bmp_array[:100, :100] = 255 * 0.5
gc.save("rect2.bmp")
