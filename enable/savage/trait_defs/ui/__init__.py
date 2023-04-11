# Import hooks for loading enable.savage.trait_defs.ui.qt.* in place of 
# a enable.savage.trait_defs.ui.qt4.* This is just the implementation, 
# it is not connected in this module, but is available for applications 
# which want to install it themselves.
#
# To use manually:
#
#     import sys
#     sys.meta_path.append(ShadowedModuleFinder())

from importlib import import_module
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
import sys


class ShadowedModuleLoader(Loader):
    """This loads another module into sys.modules with a given name.

    Parameters
    ----------
    fullname : str
        The full name of the module we're trying to import.
        Eg. "enable.savage.trait_defs.ui.qt4.foo"
    new_name : str
        The full name of the corresponding "real" module.
        Eg. "enable.savage.trait_defs.ui.qt.foo"
    new_spec : ModuleSpec
        The spec object for the corresponding "real" module.
    """

    def __init__(self, fullname, new_name, new_spec):
        self.fullname = fullname
        self.new_name = new_name
        self.new_spec = new_spec

    def create_module(self, spec):
        """Create the module object.

        This doesn't create the module object directly, rather it gets the
        underlying "real" module's object, importing it if needed.  This object
        is then returned as the "new" module.
        """
        if self.new_name not in sys.modules:
            import_module(self.new_name)
        return sys.modules[self.new_name]

    def exec_module(self, module):
        """Execute code for the module.

        This is given a module which has already been executed, so we don't
        need to execute anything.  However we do need to remove the __spec__
        that the importlibs machinery has injected into the module and
        replace it with the original spec for the underlying "real" module.
        """
        # patch up the __spec__ with the true module's original __spec__
        if self.new_spec:
            module.__spec__ = self.new_spec
            self.new_spec = None


class ShadowedModuleFinder(MetaPathFinder):
    """MetaPathFinder for shadowing modules in a package

    This finds modules with names that match a package but arranges loading
    from a different package.
    The end result is that sys.modules has two entries for pointing to the
    same module object.

    Parameters
    ----------
    package : str
        The prefix of the "shadow" package.
    true_package : str
        The prefix of the "real" package which contains the actual code.
    """

    def __init__(self, package="enable.savage.trait_defs.ui.qt.", true_package="enable.savage.trait_defs.ui.qt4."):
        self.package = package
        self.true_package = true_package

    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(self.package):
            new_name = fullname.replace(self.package, self.true_package, 1)
            new_spec = find_spec(new_name)
            if new_spec is None:
                return None
            return ModuleSpec(
                name=fullname,
                loader=ShadowedModuleLoader(fullname, new_name, new_spec),
                is_package=(new_spec.submodule_search_locations is not None),
            )
