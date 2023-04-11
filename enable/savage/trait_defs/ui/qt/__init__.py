# (C) Copyright 2005-2023 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

import os
import sys
import warnings

from traits.etsconfig.api import ETSConfig

from enable.savage.trait_defs.ui import ShadowedModuleFinder

if (
    os.environ.get('ETS_TOOLKIT', None) == "qt"  # environment says old qt4
    or ETSConfig.toolkit == "qt"  # the ETSConfig toolkit says old qt4
):
    sys.meta_path.append(ShadowedModuleFinder())

    # Importing from enable.savage.trait_defs.ui.qt4.* is deprecated
    warnings.warn(
        """The enable.savage.trait_defs.ui.qt4.* modules have moved to
        enable.savage.trait_defs.ui.qt.*

        Backward compatibility import hooks have been automatically applied.
        They will be removed in a future release of Pyface.
        """,
        DeprecationWarning,
        stacklevel=2,
    )
else:
    # Don't import from this module, use a future warning as we want end-users
    # of ETS apps to see the hints about environment variables.
    warnings.warn(
        """The enable.savage.trait_defs.ui.qt4.* modules have moved to
        enable.savage.trait_defs.ui.qt.*.

        Applications which require backwards compatibility can either:
        - set the ETS_TOOLKIT environment variable to "qt4",
        - the ETSConfig.toolkit to "qt4"
        - install pyface.ui.ShadowedModuleFinder() into sys.meta_path
        """,
        FutureWarning,
        stacklevel=2,
    )
