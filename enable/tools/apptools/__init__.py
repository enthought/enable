import warnings

warnings.warn(
    ("apptools.undo is deprecated and will be removed in a future release. The"
     " functionality is now available via pyface.undo. As a result,"
     " enable.tools.apptools has been deprecated in favor of"
     " enable.tools.pyface."),
    DeprecationWarning,
    stacklevel=2
)
