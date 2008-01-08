import setuptools
from numpy.distutils.core import setup
from setup_data import INFO


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


# Build the full set of packages by appending any found by setuptools'
# find_packages to those discovered by numpy.distutils.
config = configuration().todict()
packages = setuptools.find_packages(exclude=config['packages'] +
    ['docs', 'examples'])
config['packages'] += packages


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
    extras_require = INFO['extras_require'],
    include_package_data = True,
    install_requires = INFO['install_requires'],
    license = 'BSD',
    name = INFO['name'],
    namespace_packages = [
        "enthought",
        ],
    tests_require = [
        'nose >= 0.9',
        ],
    test_suite = 'nose.collector',
    url = 'http://code.enthought.com/ets',
    version = INFO['version'],
    zip_safe = False,
    **config
    )

