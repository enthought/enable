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
import re
import shutil
import subprocess

from numpy.distutils.core import setup

MAJOR = 4
MINOR = 5
MICRO = 1

IS_RELEASED = True

VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env,
        ).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'describe', '--tags'])
    except OSError:
        out = ''

    git_description = out.strip().decode('ascii')
    expr = r'.*?\-(?P<count>\d+)-g(?P<hash>[a-fA-F0-9]+)'
    match = re.match(expr, git_description)
    if match is None:
        git_revision, git_count = 'Unknown', '0'
    else:
        git_revision, git_count = match.group('hash'), match.group('count')

    return git_revision, git_count


def write_version_py(filename):
    template = """\
# THIS FILE IS GENERATED FROM ENABLE SETUP.PY
version = '{version}'
full_version = '{full_version}'
git_revision = '{git_revision}'
is_released = {is_released}

if not is_released:
    version = full_version
"""
    # Adding the git rev number needs to be done inside
    # write_version_py(), otherwise the import of kiva._version messes
    # up the build under Python 3.
    fullversion = VERSION
    if os.path.exists('.git'):
        git_revision, dev_num = git_version()
    elif os.path.exists('kiva/_version.py'):
        # must be a source distribution, use existing version file
        try:
            from kiva._version import git_revision, full_version
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing "
                              "kiva/_version.py and the build directory "
                              "before building.")

        match = re.match(r'.*?\.dev(?P<dev_num>\d+)', full_version)
        if match is None:
            dev_num = '0'
        else:
            dev_num = match.group('dev_num')
    else:
        git_revision = 'Unknown'
        dev_num = '0'

    if not IS_RELEASED:
        fullversion += '.dev{0}'.format(dev_num)

    with open(filename, "wt") as fp:
        fp.write(template.format(version=VERSION,
                                 full_version=fullversion,
                                 git_revision=git_revision,
                                 is_released=IS_RELEASED))


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


if __name__ == "__main__":
    write_version_py(filename='enable/_version.py')
    write_version_py(filename='kiva/_version.py')
    from enable import __version__, __requires__

    # Build the full set of packages by appending any found by setuptools'
    # find_packages to those discovered by numpy.distutils.
    config = configuration().todict()
    packages = setuptools.find_packages(exclude=config['packages'] +
                                        ['docs', 'examples'])
    packages += ['enable.savage.trait_defs.ui.wx.data']
    config['packages'] += packages

    setup(name='enable',
          version=__version__,
          author='Enthought, Inc',
          author_email='info@enthought.com',
          maintainer='ETS Developers',
          maintainer_email='enthought-dev@enthought.com',
          url='https://github.com/enthought/enable/',
          classifiers=[c.strip() for c in """\
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
          cmdclass={
              # Work around a numpy distutils bug by forcing the use of the
              # setuptools' sdist command.
              'sdist': setuptools.command.sdist.sdist,
              # Use our customized commands
              'clean': MyClean,
          },
          description='low-level drawing and interaction',
          long_description=open('README.rst').read(),
          # Note that this URL is only valid for tagged releases.
          download_url=('https://github.com/enthought/enable/archive/'
                        '{0}.tar.gz'.format(__version__)),
          install_requires=__requires__,
          license='BSD',
          package_data = {
              '': ['*.zip', '*.svg', 'images/*'],
              'enable': ['tests/primitives/data/PngSuite/*.png'],
          },
          platforms=["Windows", "Linux", "Mac OS-X", "Unix", "Solaris"],
          zip_safe=False,
          **config)
