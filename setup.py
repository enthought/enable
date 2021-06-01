# (C) Copyright 2008-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

from distutils.command.build import build as base_build
import glob
import os
import re
import runpy
import subprocess
import sys

from Cython.Distutils import build_ext
import numpy
from setuptools import Extension, find_packages, setup
from setuptools.command.install import install as base_install

MAJOR = 5
MINOR = 2
MICRO = 0
PRERELEASE = "rc2"
IS_RELEASED = False

# Templates for version strings.
RELEASED_VERSION = "{major}.{minor}.{micro}{prerelease}"
UNRELEASED_VERSION = "{major}.{minor}.{micro}{prerelease}.dev{dev}"


# Distutils will collect the `.py` files for the distribution before
# building the Extensions, so `.py` files generated by SWIG will not be
# added, resulting in a broken build.
# See: https://stackoverflow.com/a/21236111

class PatchedBuild(base_build):
    def run(self):
        self.run_command('build_ext')
        base_build.run(self)


class PatchedInstall(base_install):
    def run(self):
        self.run_command('build_ext')
        super().run()


def git_version():
    """ Return the git revision as a string
    """
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

#: The full version of the package, including a development suffix
#: for unreleased versions of the package.
version = "{version}"

#: The full version of the package, same as 'version'
#: Kept for backward compatibility
full_version = version

#: The Git revision from which this release was made.
git_revision = "{git_revision}"

#: Flag whether this is a final release
is_released = {is_released}
"""
    # Adding the git rev number needs to be done inside
    # write_version_py(), otherwise the import of kiva._version messes
    # up the build under Python 3.
    version_template = RELEASED_VERSION if IS_RELEASED else UNRELEASED_VERSION
    kiva_version_path = os.path.join(os.path.dirname(__file__), 'kiva',
                                     '_version.py')
    if os.path.exists(os.path.join(os.path.dirname(__file__), '.git')):
        git_revision, dev_num = git_version()
    elif os.path.exists(kiva_version_path):
        # must be a source distribution, use existing version file
        try:
            context = runpy.run_path(kiva_version_path)
            git_revision, version = context['git_revision'], context['version']
        except (SyntaxError, KeyError):
            raise RuntimeError("Unable to read git_revision. Try removing "
                               "kiva/_version.py and the build directory "
                               "before building.")

        match = re.match(r'.*?\.dev(?P<dev_num>\d+)', version)
        if match is None:
            dev_num = '0'
        else:
            dev_num = match.group('dev_num')
    else:
        git_revision = 'Unknown'
        dev_num = '0'

    with open(filename, "wt") as fp:
        fp.write(
            template.format(
                version=version_template.format(
                    major=MAJOR,
                    minor=MINOR,
                    micro=MICRO,
                    prerelease=PRERELEASE,
                    dev=dev_num
                ),
                git_revision=git_revision,
                is_released=IS_RELEASED
            )
        )


def agg_extensions():
    kiva_agg_dir = os.path.join('kiva', 'agg')
    agg_dir = os.path.join(kiva_agg_dir, 'agg-24')
    freetype_dir = os.path.join(kiva_agg_dir, 'freetype2')
    freetype2_sources = [
        'autofit/autofit.c', 'base/ftbase.c', 'base/ftbbox.c', 'base/ftbdf.c',
        'base/ftbitmap.c', 'base/ftdebug.c', 'base/ftglyph.c', 'base/ftinit.c',
        'base/ftmm.c', 'base/ftsystem.c', 'base/fttype1.c', 'base/ftxf86.c',
        'bdf/bdf.c', 'cff/cff.c', 'cid/type1cid.c', 'gzip/ftgzip.c',
        'lzw/ftlzw.c', 'pcf/pcf.c', 'pfr/pfr.c', 'psaux/psaux.c',
        'pshinter/pshinter.c', 'psnames/psnames.c', 'raster/raster.c',
        'sfnt/sfnt.c', 'smooth/smooth.c', 'truetype/truetype.c',
        'type1/type1.c', 'type42/type42.c', 'winfonts/winfnt.c',
    ]
    freetype2_dirs = [
        'autofit', 'base', 'bdf', 'cache', 'cff', 'cid', 'gxvalid', 'gzip',
        'lzw', 'otvalid', 'pcf', 'pfr', 'psaux', 'pshinter', 'psnames',
        'raster', 'sfnt', 'smooth', 'tools', 'truetype', 'type1', 'type42',
        'winfonts',
    ]

    define_macros = [
        # Numpy defines
        ('NUMPY', None),
        ('PY_ARRAY_TYPES_PREFIX', 'NUMPY_CXX'),
        ('OWN_DIMENSIONS', '0'),
        ('OWN_STRIDES', '0'),
        # Freetype defines
        ('FT2_BUILD_LIBRARY', None)
    ]
    extra_link_args = []
    include_dirs = []

    if sys.platform == 'win32':
        plat = 'win32'
    elif sys.platform == 'darwin':
        plat = 'osx'
        darwin_frameworks = ['ApplicationServices']
        for framework in darwin_frameworks:
            extra_link_args.extend(['-framework', framework])

        include_dirs += [
            '/System/Library/Frameworks/%s.framework/Versions/A/Headers' % x
            for x in darwin_frameworks
        ]
        define_macros += [('__DARWIN__', None)]
    else:
        # This should work for most linux distributions
        plat = 'x11'

    freetype2_sources = [os.path.join(freetype_dir, 'src', src)
                         for src in freetype2_sources]
    freetype2_dirs = [
        os.path.join(freetype_dir, 'src'),
        os.path.join(freetype_dir, 'include'),
    ] + [os.path.join(freetype_dir, 'src', d) for d in freetype2_dirs]

    agg_sources = [
        *glob.glob(os.path.join(agg_dir, 'src', '*.cpp')),
        *glob.glob(os.path.join(agg_dir, 'font_freetype', '*.cpp')),
    ]
    kiva_agg_sources = [
        *glob.glob(os.path.join(kiva_agg_dir, 'src', 'kiva_*.cpp')),
    ] + agg_sources + freetype2_sources
    agg_include_dirs = [
        os.path.join(agg_dir, 'include'),
        os.path.join(agg_dir, 'font_freetype'),
    ] + freetype2_dirs
    include_dirs += [
        numpy.get_include(),
        os.path.join(kiva_agg_dir, 'src'),
    ] + agg_include_dirs
    swig_opts = [
        '-I' + os.path.join(kiva_agg_dir, 'src'),
        '-I' + os.path.join(agg_dir, 'include'),
        '-c++',
    ]

    # Platform support extension
    plat_support_sources = [
        os.path.join(kiva_agg_dir, 'src', plat, 'plat_support.i'),
        os.path.join(kiva_agg_dir, 'src', plat, 'agg_bmp.cpp'),
    ]
    plat_support_swig_opts = [
        '-outdir', kiva_agg_dir,  # write plat_support.py to this dir
        '-c++',
        '-I' + os.path.join(kiva_agg_dir, 'src'),
    ]
    plat_support_libraries = []
    if plat != 'osx':
        plat_support_sources.append(
            os.path.join(kiva_agg_dir, 'src', plat,
                         'agg_platform_specific.cpp')
        )
    if plat == 'win32':
        plat_support_libraries += ['gdi32', 'user32']
    elif plat == 'x11':
        plat_support_libraries += ['X11']

    return [
        Extension(
            'kiva.agg._agg',
            sources=[
                os.path.join(kiva_agg_dir, 'agg.i'),
            ] + kiva_agg_sources,
            swig_opts=swig_opts,
            include_dirs=include_dirs,
            extra_link_args=extra_link_args,
            define_macros=define_macros,
            language='c++',
        ),
        Extension(
            'kiva.agg._plat_support',
            sources=plat_support_sources,
            swig_opts=plat_support_swig_opts,
            include_dirs=include_dirs,
            extra_link_args=extra_link_args,
            define_macros=define_macros,
            libraries=plat_support_libraries,
            language='c++',
        ),
    ]


def base_extensions():
    return [
        Extension(
            'kiva._cython_speedups',
            sources=[
                'kiva/_cython_speedups.pyx',
                'kiva/_hit_test.cpp'
            ],
            depends=[
                'kiva/_hit_test.h',
                'kiva/_hit_test.pxd',
            ],
            include_dirs=['kiva', numpy.get_include()],
            language='c++',
        ),
        Extension(
            'kiva._marker_renderer',
            sources=['kiva/_marker_renderer.pyx'],
            depends=[
                'kiva/_marker_renderer.pxd',
            ],
            include_dirs=[
                os.path.join('kiva', 'markers', 'agg'),
                os.path.join('kiva', 'markers'),
                numpy.get_include(),
            ],
            define_macros=[
                ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ],
            language='c++',
        ),
    ]


def gl_extensions():
    kiva_gl_dir = os.path.join('kiva', 'gl')
    agg_dir = os.path.join(kiva_gl_dir, 'src', 'agg')

    kiva_gl_libraries = []
    define_macros = []
    extra_compile_args = []
    extra_link_args = []
    include_dirs = []

    if sys.platform == 'win32':
        kiva_gl_libraries += ['opengl32', 'glu32']
    elif sys.platform == 'darwin':
        # Options to make macOS link OpenGL
        darwin_frameworks = ['ApplicationServices', 'OpenGL']
        for framework in darwin_frameworks:
            extra_link_args.extend(['-framework', framework])

        include_dirs += [
            '/System/Library/Frameworks/%s.framework/Versions/A/Headers' % x
            for x in darwin_frameworks
        ]
        define_macros += [
            ('__DARWIN__', None),
            # OpenGL is deprecated starting with macOS 10.14 and gone in 10.15
            # But that doesn't mean we want to hear about it. We know, Apple.
            ('GL_SILENCE_DEPRECATION', None),
        ]
        extra_compile_args = [
           '-Wfatal-errors',
           '-Wno-unused-function',
        ]
    else:
        # This should work for most linux distributions
        kiva_gl_libraries += ['GL', 'GLU']
        extra_compile_args = [
           '-Wfatal-errors',
           '-Wno-unused-function',
        ]

    kiva_gl_sources = [
        *glob.glob(os.path.join(kiva_gl_dir, 'src', 'kiva_gl_*.cpp')),
        *glob.glob(os.path.join(agg_dir, '*.cpp')),
    ]
    include_dirs += [
        os.path.join(kiva_gl_dir, 'src'),
        agg_dir,
        numpy.get_include(),
    ]
    swig_opts = [
        '-I' + os.path.join(kiva_gl_dir, 'src', 'swig'),
        '-I' + os.path.join(kiva_gl_dir, 'src'),
        '-I' + agg_dir,
        '-c++',
    ]

    return [
        Extension(
            'kiva.gl._gl',
            sources=[
                os.path.join(kiva_gl_dir, 'gl.i'),
            ] + kiva_gl_sources,
            swig_opts=swig_opts,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=define_macros,
            libraries=kiva_gl_libraries,
            language='c++',
        ),
    ]


def macos_extensions():
    extra_link_args = []
    frameworks = [
        'Cocoa', 'CoreFoundation', 'ApplicationServices', 'Foundation'
    ]
    include_dirs = [
        '/System/Library/Frameworks/%s.framework/Versions/A/Headers' % x
        for x in frameworks
    ]
    for framework in frameworks:
        extra_link_args.extend(['-framework', framework])

    return [
        Extension(
            'kiva.quartz.ABCGI',
            sources=[
                'kiva/quartz/ABCGI.pyx',
                'kiva/quartz/Python.pxi',
                'kiva/quartz/numpy.pxi',
                'kiva/quartz/c_numpy.pxd',
                'kiva/quartz/CoreFoundation.pxi',
                'kiva/quartz/CoreGraphics.pxi',
                'kiva/quartz/CoreText.pxi',
            ],
            extra_link_args=extra_link_args,
            include_dirs=[numpy.get_include()],
        ),
        Extension(
            'kiva.quartz.CTFont',
            sources=[
                'kiva/quartz/CTFont.pyx',
                'kiva/quartz/CoreFoundation.pxi',
                'kiva/quartz/CoreGraphics.pxi',
                'kiva/quartz/CoreText.pxi',
            ],
            extra_link_args=extra_link_args,
        ),
        Extension(
            'kiva.quartz.mac_context',
            sources=[
                'kiva/quartz/mac_context.c',
                'kiva/quartz/mac_context_cocoa.m',
            ],
            depends=[
                'kiva/quartz/mac_context.h',
            ],
            extra_link_args=extra_link_args,
            include_dirs=include_dirs,
        )
    ]


def verify_swig_version():
    """ Verify we are using swiig version 3.0.x.
    """
    msg = ("SWIG is a required build dependency of Enable. Furthermore, there "
           "is currently a known issue with SWIG 4.0, "
           "see enthought/enable#360. Please install SWIG 3.0.x "
           "(see http://www.swig.org/).")
    try:
        cmd = ["swig", "-version"]
        opts = {"stdout": subprocess.PIPE, "encoding": "utf-8"}
        with subprocess.Popen(cmd, **opts) as proc:
            # We expect text along the lines of "SWIG Version 3.X.Y"
            swig_version_match = re.search(
                r"(SWIG Version 3)\.\d{1,2}\.\d{1,2}",
                proc.stdout.read()
            )
            if swig_version_match is None:
                raise Exception(msg)
    except FileNotFoundError:
        raise Exception(msg)


if __name__ == "__main__":
    verify_swig_version()

    # Write version modules as needed
    enable_version_path = os.path.join('enable', '_version.py')
    write_version_py(filename=enable_version_path)
    write_version_py(filename=os.path.join('kiva', '_version.py'))

    # Read the version
    context = runpy.run_path(enable_version_path)
    __version__ = context['full_version']

    # Read the requirements from enable.__init__
    enable_init_path = os.path.join('enable', '__init__.py')
    context = runpy.run_path(enable_init_path)
    __extras_require__ = context['__extras_require__']
    __requires__ = context['__requires__']

    with open('README.rst', 'r') as fp:
        long_description = fp.read()

    # Collect extensions
    ext_modules = base_extensions() + agg_extensions() + gl_extensions()
    if sys.platform == 'darwin':
        ext_modules += macos_extensions()

    setup(name='enable',
          version=__version__,
          author='Enthought, Inc',
          author_email='info@enthought.com',
          maintainer='ETS Developers',
          maintainer_email='enthought-dev@enthought.com',
          url='https://github.com/enthought/enable/',
          # Note that this URL is only valid for tagged releases.
          download_url=('https://github.com/enthought/enable/archive/'
                        '{0}.tar.gz'.format(__version__)),
          license='BSD',
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
          platforms=['Windows', 'Linux', 'macOS', 'Unix', 'Solaris'],
          description='low-level drawing and interaction',
          long_description=long_description,
          install_requires=__requires__,
          extras_require=__extras_require__,
          cmdclass={
              'build': PatchedBuild,
              'install': PatchedInstall,
              'build_ext': build_ext,
          },
          entry_points={
            'enable.toolkits': [
                'null = enable.null.toolkit:toolkit',
                'qt = enable.qt4.toolkit:toolkit',
                'qt4 = enable.qt4.toolkit:toolkit',
                'wx = enable.wx.toolkit:toolkit',
            ],
            'etsdemo_data': [
                'enable_examples = enable.examples._etsdemo_info:info',
                'kiva_examples = kiva.examples._etsdemo_info:info',
            ]
          },
          ext_modules=ext_modules,
          packages=find_packages(exclude=['ci', 'docs']),
          package_data={
              '': ['*.zip', '*.svg', 'images/*'],
              'enable': ['tests/primitives/data/PngSuite/*.png'],
              'enable.examples': ['demo/*',
                                  'demo/*/*',
                                  'demo/*/*/*',
                                  'demo/*/*/*/*',
                                  'demo/*/*/*/*/*'],
              'enable.savage.trait_defs.ui.wx': ['data/*.svg'],
              'kiva': ['tests/agg/doubleprom_soho_full.jpg',
                       'fonttools/tests/data/*.afm',
                       'fonttools/tests/data/*.ttc',
                       'fonttools/tests/data/*.ttf',
                       'fonttools/tests/data/*.txt'],
              'kiva.examples': ['kiva/*',
                                'kiva/*/*'],
          },
          zip_safe=False,
          )
