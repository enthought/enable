#
# (C) Copyright 2015 Enthought, Inc., Austin, TX
# All right reserved.
#
# This file is open source software distributed according to the terms in
# LICENSE.txt
#
import warnings

warnings.warn(
    (
        "apptools.undo is deprecated and will be removed in a future release. "
        "The functionality is now available via pyface.undo. As a result, "
        "enable.tools.apptools has been deprecated in favor of "
        "enable.tools.pyface."
    ),
    DeprecationWarning,
    stacklevel=2,
)
