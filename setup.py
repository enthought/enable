# Copyright (c) 2008-2013 by Enthought, Inc.
# All rights reserved.

# These are necessary to get the clib compiled.  The following also adds
# an additional option --compiler=STR to develop, which usually does not
# have such an option.  The code below is a bad hack, as it changes
# sys.argv to fool setuptools which therefore has to be imported BELOW
# this hack.
import sys
if 'develop' in sys.argv:
    idx = sys.argv.index('develop')
    compiler = []
    for arg in sys.argv[idx+1:]:
        if arg.startswith('--compiler='):
            compiler = ['-c', arg[11:]]
            del sys.argv[idx+1:]
    # insert extra options right before 'develop'
    sys.argv[idx:idx] = ['build_src', '--inplace', 'build_clib'] + compiler + \
        ['build_ext', '--inplace'] + compiler

from os.path import join

# Setuptools must be imported BEFORE numpy.distutils for things to work right!
import setuptools

import distutils
import distutils.command.clean
import os
import shutil

from numpy.distutils.core import setup


info = {}
execfile(join('enable', '__init__.py'), info)


# Configure python extensions.
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )

    config.add_subpackage('kiva')

    return config



# Build the full set of packages by appending any found by setuptools'
# find_packages to those discovered by numpy.distutils.
config = configuration().todict()
packages = setuptools.find_packages(exclude=config['packages'] +
                                    ['docs', 'examples'])
packages += ['enable.savage.trait_defs.ui.wx.data']
config['packages'] += packages


class MyClean(distutils.command.clean.clean):
    '''
    Subclass to remove any files created in an inplace build.

    This subclasses distutils' clean because neither setuptools nor
    numpy.distutils implements a clean command.

    '''
    def run(self):
        distutils.command.clean.clean.run(self)

        # Clean any build or dist directory
        if os.path.isdir("build"):
            shutil.rmtree("build", ignore_errors=True)
        if os.path.isdir("dist"):
            shutil.rmtree("dist", ignore_errors=True)

        # Clean out any files produced by an in-place build.  Note that our
        # code assumes the files are relative to the 'kiva' dir.
        INPLACE_FILES = (
            # Common AGG
            join("agg", "agg.py"),
            join("agg", "plat_support.py"),
            join("agg", "agg_wrap.cpp"),

            # Mac
            join("quartz", "ABCGI.so"),
            join("quartz", "ABCGI.c"),
            join("quartz", "macport.so"),
            join("quartz", "mac_context.so"),
            join("quartz", "CTFont.so"),
            join("quartz", "CTFont.c"),

            # Win32 Agg
            join("agg", "_agg.pyd"),
            join("agg", "_plat_support.pyd"),
            join("agg", "src", "win32", "plat_support.pyd"),

            # *nix Agg
            join("agg", "_agg.so"),
            join("agg", "_plat_support.so"),
            join("agg", "src", "x11", "plat_support_wrap.cpp"),

            # Misc
            join("agg", "src", "gl", "plat_support_wrap.cpp"),
            join("agg", "src", "gl", "plat_support.py"),
            )
        for f in INPLACE_FILES:
            f = join("kiva", f)
            if os.path.isfile(f):
                os.remove(f)


setup(
    name = 'enable',
    version = info['__version__'],
    author = 'Enthought, Inc',
    author_email = 'info@enthought.com',
    maintainer = 'ETS Developers',
    maintainer_email = 'enthought-dev@enthought.com',
    url = 'http://code.enthought.com/projects/enable',
    classifiers = [c.strip() for c in """\
        Development Status :: 5 - Production/Stable
        Intended Audience :: Developers
        Intended Audience :: Science/Research
        License :: OSI Approved :: BSD License
        Operating System :: MacOS
        Operating System :: Microsoft :: Windows
        Operating System :: OS Independent
        Operating System :: POSIX
        Operating System :: Unix
        Programming Language :: C
        Programming Language :: Python
        Topic :: Scientific/Engineering
        Topic :: Software Development
        Topic :: Software Development :: Libraries
        """.splitlines() if len(c.strip()) > 0],
    cmdclass = {
        # Work around a numpy distutils bug by forcing the use of the
        # setuptools' sdist command.
        'sdist': setuptools.command.sdist.sdist,

        # Use our customized commands
        'clean': MyClean,
        },
    description = 'low-level drawing and interaction',
    long_description = open('README.rst').read(),
    download_url = ('http://www.enthought.com/repo/ets/enable-%s.tar.gz' %
                    info['__version__']),
    install_requires = info['__requires__'],
    license = 'BSD',
    package_data = {'': ['*.zip', '*.svg', 'images/*']},
    platforms = ["Windows", "Linux", "Mac OS-X", "Unix", "Solaris"],
    setup_requires = [
        'cython',
    ],
    zip_safe = False,
    **config
)
