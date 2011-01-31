#!/usr/bin/env python
#
# Copyright (c) 2008-2011 by Enthought, Inc.
# All rights reserved.

"""
Drawing and interaction packages.

The Enable *project* provides two related multi-platform *packages* for drawing
GUI objects.

- **Enable**: An object drawing library that supports containment and event
  notification.
- **Kiva**: A multi-platform DisplayPDF vector drawing engine.

Enable
------

The Enable package is a multi-platform object drawing library built on top of
Kiva. The core of Enable is a container/component model for drawing and event
notification. The core concepts of Enable are:

- Component
- Container
- Events (mouse, drag, and key events)

Enable provides a high-level interface for creating GUI objects, while
enabling a high level of control over user interaction. Enable is a supporting
technology for the Chaco and BlockCanvas projects.


Kiva
----
Kiva is a multi-platform DisplayPDF vector drawing engine that supports
multiple output backends, including Windows, GTK, and Macintosh native
windowing systems, a variety of raster image formats, PDF, and Postscript.

DisplayPDF is more of a convention than an actual specification. It is a
path-based drawing API based on a subset of the Adobe PDF specification.
Besides basic vector drawing concepts such as paths, rects, line sytles, and
the graphics state stack, it also supports pattern fills, antialiasing, and
transparency. Perhaps the most popular implementation of DisplayPDF is
Apple's Quartz 2-D graphics API in Mac OS X.

Kiva Features
`````````````
Kiva currently implements the following features:

- paths and compiled paths; arcs, bezier curves, rectangles
- graphics state stack
- clip stack, disjoint rectangular clip regions
- raster image blitting
- arbitrary affine transforms of the graphics context
- bevelled and mitered joins
- line width, line dash
- Freetype or native fonts
- RGB, RGBA, or grayscale color depths
- transparency

Prerequisites
-------------

You must have the following libraries installed before building or installing
the Enable project:

* `setuptools <http://pypi.python.org/pypi/setuptools/0.6c8>`_
* `SWIG <http://www.swig.org/>`_ version 1.3.30 or later.
* `Cython <http://www.cython.org>`_ version 0.13 or later
* `Numpy <http://pypi.python.org/pypi/numpy/1.3.1>`_  version 1.3.1
* `ReportLab Toolkit <http://www.reportlab.org/rl_toolkit.html/>`_ for PDF
  backend support in Kiva.
"""


# FIXME: Hack to add arguments to setup.py develop
#
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


# Setuptools must be imported BEFORE numpy.distutils for things to work right!
import setuptools

import distutils
import os
import shutil

from numpy.distutils.core import setup


# FIXME: This works around a setuptools bug which gets setup_data.py metadata
# from incorrect packages. Ticket #1592
#from setup_data import INFO
setup_data = dict(__name__='', __file__='setup_data.py')
execfile('setup_data.py', setup_data)
INFO = setup_data['INFO']


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

    config.add_subpackage('enthought.kiva')
    config.add_subpackage('enthought')

    return config



# Build the full set of packages by appending any found by setuptools'
# find_packages to those discovered by numpy.distutils.
config = configuration().todict()
packages = setuptools.find_packages(exclude=config['packages'] +
    ['docs', 'examples'])
packages += ['enthought.savage.traits.ui.wx.data']
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
        # code assumes the files are relative to the 'enthought/kiva' dir.
        INPLACE_FILES = (
            # Common AGG
            os.path.join("agg", "agg.py"),
            os.path.join("agg", "plat_support.py"),
            os.path.join("agg", "agg_wrap.cpp"),

            # Mac
            os.path.join("quartz", "ABCGI.so"),
            os.path.join("quartz", "macport.so"),
            os.path.join("quartz", "ABCGI.c"),
            os.path.join("quartz", "ATSFont.so"),
            os.path.join("quartz", "ATSFont.c"),

            # Win32 Agg
            os.path.join("agg", "_agg.pyd"),
            os.path.join("agg", "_plat_support.pyd"),
            os.path.join("agg", "src", "win32", "plat_support.pyd"),

            # *nix Agg
            os.path.join("agg", "_agg.so"),
            os.path.join("agg", "_plat_support.so"),
            os.path.join("agg", "src", "x11", "plat_support_wrap.cpp"),

            # Misc
            os.path.join("agg", "src", "gl", "plat_support_wrap.cpp"),
            os.path.join("agg", "src", "gl", "plat_support.py"),
            )
        for f in INPLACE_FILES:
            f = os.path.join("enthought", "kiva", f)
            if os.path.isfile(f):
                os.remove(f)


# The actual setup call.
DOCLINES = __doc__.split("\n")
setup(
    author = 'Enthought, Inc',
    author_email = 'info@enthought.com',
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
    description = DOCLINES[1],
    download_url = ('http://www.enthought.com/repo/ETS/Enable-%s.tar.gz' %
                    INFO['version']),
    include_package_data = True,
    install_requires = INFO['install_requires'],
    license = 'BSD',
    long_description = '\n'.join(DOCLINES[3:]),
    maintainer = 'ETS Developers',
    maintainer_email = 'enthought-dev@enthought.com',
    name = INFO['name'],
    namespace_packages = [
        "enthought",
        ],
    package_data = {'': ['*.zip', '*.svg']},
    platforms = ["Windows", "Linux", "Mac OS-X", "Unix", "Solaris"],
    setup_requires = [
        'cython',
        ],
    tests_require = [
        'nose >= 0.10.3',
        ],
    test_suite = 'nose.collector',
    url = 'http://code.enthought.com/projects/enable',
    version = INFO['version'],
    zip_safe = False,
    **config
)
