
from unittest import mock


def patch_global_font_manager(new_value):
    """ Patch the global FontManager instance at the module level.

    Useful for avoiding test interaction due to the global font manager
    cache being created at runtime.

    Parameters
    ----------
    new_value : FontManager or None
        Temporary value to be used as the global font manager.

    Returns
    -------
    patcher : unittest.mock._patch
    """
    return mock.patch("kiva.fonttools.font_manager.fontManager", new_value)
