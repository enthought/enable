from setuptools import find_packages
from numpy.distutils.core import setup


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )

    config.add_subpackage('enthought.kiva')
    config.add_data_files('enthought/__init__.py')

    return config


# The following monkeypatching code comes from Numpy distutils.
import numpy

if 1:  # numpy.__version__[:5] < '1.0.3':
    # numpy 1.0.3 provides a fix to distutils to make sure that build_clib is
    # run before build_ext. This is critical for semi-automatic building of
    # many extension modules using numpy. Here, we monkey-patch the run method
    # of the build_ext command to provide the fix in 1.0.3.

    from numpy.distutils.command import build_ext
    old_run = build_ext.build_ext.run

    def new_run(self):
        if not self.extensions:
            return

        # Make sure that extension sources are complete.
        self.run_command('build_src')

        if self.distribution.has_c_libraries():
            self.run_command('build_clib')
            build_clib = self.get_finalized_command('build_clib')
            self.library_dirs.append(build_clib.build_clib)
        else:
            build_clib = None

        old_run(self)

    build_ext.build_ext.run = new_run


# Function to convert simple ETS project names and versions to a requirements
# spec that works for both development builds and stable builds.  Allows
# a caller to specify a max version, which is intended to work along with
# Enthought's standard versioning scheme -- see the following write up:
#    https://svn.enthought.com/enthought/wiki/EnthoughtVersionNumbers
def etsdep(p, min, max=None, literal=False):
    require = '%s >=%s.dev' % (p, min)
    if max is not None:
        if literal is False:
            require = '%s, <%s.a' % (require, max)
        else:
            require = '%s, <%s' % (require, max)
    return require


# Declare our ETS project dependencies.
ENTHOUGHTBASE = etsdep('EnthoughtBase', '3.0.0b1')
TRAITSBACKENDWX = etsdep('TraitsBackendWX', '3.0.0b1')
TRAITSBACKENDQT = etsdep('TraitsBackendQt', '3.0.0b1')
TRAITS_UI = etsdep('Traits[ui]', '3.0.0b1')


setup(
    author = 'Enthought, Inc',
    author_email = 'info@enthought.com',
    cmdclass = {
        # Work around a numpy distutils bug by forcing the use of the
        # setuptools' sdist command.
        'sdist': setuptools.command.sdist.sdist,
        },
    dependency_links = [
        'http://code.enthought.com/enstaller/eggs/source',
        ],
    description = ('Multi-platform vector drawing engine that supports '
        'multiple output backends'),
    extras_require = {
        'ps': [],
        'svg': [],
        'traits': [
            ],
        'qt': [
            TRAITSBACKENDQT,
            ],
        "wx": [
            TRAITSBACKENDWX,
            ],

        # All non-ets dependencies should be in this extra to ensure users can
        # decide whether to require them or not.
        'nonets': [
            "numpy >=1.0.2",
            ],
        },
    include_package_data = True,
    install_requires = [
        ENTHOUGHTBASE,
        TRAITS_UI,
        ],
    license = 'BSD',
    name = 'Enable',
    namespace_packages = [
        "enthought",
        ],
    packages = find_packages(
        exclude=['examples'],
        ),
    tests_require = [
        'nose >= 0.9',
        ],
    test_suite = 'nose.collector',
    url = 'http://code.enthought.com/ets',
    version = '3.0.0b1',
    zip_safe = False,
    **configuration().todict()
    )

