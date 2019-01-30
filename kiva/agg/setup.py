#!/usr/bin/env python
import sys
import os
import re
import platform

freetype2_sources =['autofit/autofit.c',
                    'base/ftbase.c','base/ftsystem.c','base/ftinit.c',
                    'base/ftglyph.c','base/ftmm.c','base/ftbdf.c',
                    'base/ftbbox.c','base/ftdebug.c','base/ftxf86.c',
                    'base/fttype1.c',
                    'bdf/bdf.c',
                    'cff/cff.c',
                    'cid/type1cid.c',
                    'lzw/ftlzw.c',
                    'pcf/pcf.c','pfr/pfr.c',
                    'psaux/psaux.c',
                    'pshinter/pshinter.c',
                    'psnames/psnames.c',
                    'raster/raster.c',
                    'sfnt/sfnt.c',
                    'smooth/smooth.c',
                    'truetype/truetype.c',
                    'type1/type1.c',
                    'type42/type42.c',
                    'winfonts/winfnt.c',
                    'gzip/ftgzip.c',
                    'base/ftmac.c',
                    ]

freetype2_dirs = [
    'autofit',
    'base',
    'bdf',
    'cache',
    'cff',
    'cid',
    'gxvalid',
    'gzip',
    'lzw',
    'otvalid',
    'pcf',
    'pfr',
    'psaux',
    'pshinter',
    'psnames',
    'raster',
    'sfnt',
    'smooth',
    'tools',
    'truetype',
    'type1',
    'type42',
    'winfonts',
    'gzip'
    ]



def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import dict_append, get_info

    agg_dir = 'agg-24'
    agg_lib = 'agg24_src'

    config = Configuration('agg', parent_package,top_path)
    numerix_info = get_info('numerix')

    if ('NUMPY', None) in numerix_info.get('define_macros',[]):
        dict_append(numerix_info,
                    define_macros = [('PY_ARRAY_TYPES_PREFIX','NUMPY_CXX'),
                                     ('OWN_DIMENSIONS','0'),
                                     ('OWN_STRIDES','0')])

    #-------------------------------------------------------------------------
    # Configure the Agg backend to use on each platform
    #-------------------------------------------------------------------------
    if sys.platform=='win32':
        plat = 'win32'
    elif sys.platform == 'darwin':
        plat = 'gl'
    else:
        #plat = 'gtk1'  # use with gtk1, it's fast
        plat = 'x11'  # use with gtk2, it's slow but reliable
        #plat = 'gdkpixbuf2'


    #-------------------------------------------------------------------------
    # Add the freetype library (agg 2.4 links against this)
    #-------------------------------------------------------------------------

    prefix = config.paths('freetype2/src')[0]
    freetype_lib = 'freetype2_src'

    def get_ft2_sources(name_info, build_dir):
        (lib_name, build_info) = name_info
        sources = [prefix + "/" + s for s in freetype2_sources]
        if sys.platform=='darwin':
            return sources[:]
        return sources[:-1]

    ft2_incl_dirs = ['freetype2/src/' + s for s in freetype2_dirs] \
                    + ['freetype2/include', 'freetype2/src']
    ft2_incl_dirs = config.paths(*ft2_incl_dirs)
    if sys.platform == 'darwin' and '64bit' not in platform.architecture():
        ft2_incl_dirs.append("/Developer/Headers/FlatCarbon")

    config.add_library(freetype_lib,
                       sources = [get_ft2_sources],
                       include_dirs = ft2_incl_dirs,

                       # This macro was introduced in Freetype 2.2; if it is
                       # not defined, then the ftheader.h file (one of the
                       # primary headers) won't pull in any additional internal
                       # Freetype headers, and the library will mysteriously
                       # fail to build.
                       macros = [("FT2_BUILD_LIBRARY", None)],

                       depends = ['freetype2'],
                       )

    #-------------------------------------------------------------------------
    # Add the Agg sources
    #-------------------------------------------------------------------------

    agg_include_dirs = [agg_dir+'/include',agg_dir+'/font_freetype'] + \
                                   ft2_incl_dirs
    agg_sources = [agg_dir+'/src/*.cpp',
                    agg_dir+'/font_freetype/*.cpp']
    config.add_library(agg_lib,
                       agg_sources,
                       include_dirs = agg_include_dirs,
                       depends = [agg_dir])

    #-------------------------------------------------------------------------
    # Add the Kiva sources
    #-------------------------------------------------------------------------
    if sys.platform == 'darwin':
        define_macros = [('__DARWIN__', None)]
        macros = [('__DARWIN__', None)]
    else:
        define_macros = []
        macros = []

    kiva_include_dirs = ['src'] + agg_include_dirs
    config.add_library('kiva_src',
                       ['src/kiva_*.cpp', 'src/gl_graphics_context.cpp'],
                       include_dirs = kiva_include_dirs,
                       # Use "macros" instead of "define_macros" because the
                       # latter is only used for extensions, and not clibs
                       macros = macros,
                       )

    # MSVC6.0: uncomment to handle template parameters:
    #extra_compile_args = ['/Zm1000']
    extra_compile_args = []

    # XXX: test whether numpy has weakref support

    #-------------------------------------------------------------------------
    # Build the extension itself
    #-------------------------------------------------------------------------

    # Check for g++ < 4.0 on 64-bit Linux
    use_32bit_workaround = False

    if sys.platform == 'linux2' and '64bit' in platform.architecture():
        f = os.popen("g++ --version")
        line0 = f.readline()
        f.close()
        m = re.match(r'.+?\s([3-5])\.\d+', line0)
        if m is not None and int(m.group(1)) < 4:
            use_32bit_workaround = True

    # Enable workaround of agg bug on 64-bit machines with g++ < 4.0
    if use_32bit_workaround:
        define_macros.append(("ALWAYS_32BIT_WORKAROUND", 1))

    # Options to make OS X link OpenGL
    if '64bit' not in platform.architecture():
        darwin_frameworks = ['Carbon', 'ApplicationServices', 'OpenGL']
    else:
        darwin_frameworks = ['ApplicationServices', 'OpenGL']    

    darwin_extra_link_args = []
    for framework in darwin_frameworks:
        darwin_extra_link_args.extend(['-framework', framework])

    darwin_opengl_opts = dict(
            include_dirs = [
              '/System/Library/Frameworks/%s.framework/Versions/A/Headers' % x
              for x in darwin_frameworks],
            define_macros = [('__DARWIN__',None)],
            extra_link_args = darwin_extra_link_args
            )

    build_info = {}
    kiva_lib = 'kiva_src'
    build_libraries = [kiva_lib, agg_lib, freetype_lib]
    if sys.platform == "win32":
        build_libraries += ["opengl32", "glu32"]
    elif sys.platform == "darwin":
        dict_append(build_info, **darwin_opengl_opts)
    else:
        # This should work for most linuxes (linuces?)
        build_libraries += ["GL", "GLU"]
    dict_append(build_info,
                sources = ['agg.i'],
                include_dirs = kiva_include_dirs,
                libraries = build_libraries,
                depends = ['src/*.[ih]'],
                extra_compile_args = extra_compile_args,
                extra_link_args = [],
                define_macros=define_macros,
                )
    dict_append(build_info, **numerix_info)
    config.add_extension('_agg', **build_info)

    sources = [os.path.join('src',plat,'plat_support.i'),
               os.path.join('src',plat,'agg_bmp.cpp'),
               ]
    if plat != 'gl':
        sources.append(os.path.join('src',plat,'agg_platform_specific.cpp'))

    plat_info = {}
    dict_append(plat_info, libraries = [agg_lib],
                include_dirs = kiva_include_dirs,
                extra_compile_args = extra_compile_args,
                depends = ['src'])
    dict_append(plat_info, **numerix_info)

    if plat=='win32':
        dict_append(plat_info, libraries = ['gdi32','user32'])

    elif plat in ['x11','gtk1']:
        # Make sure we raise an error if the information is not found.
        # Frequently, the 64-bit libraries are not in a known location and need
        # manual configuration. From experience, this is usually not detected by
        # the builder if we do not raise an exception.
        x11_info = get_info('x11', notfound_action=2)
        dict_append(plat_info, **x11_info)

    elif plat=='gdkpixbuf2':
        #gdk_pixbuf_xlib_2 = get_info('gdk_pixbuf_xlib_2',notfound_action=1)
        #dict_append(plat_info,**gdk_pixbuf_xlib_2)
        gtk_info = get_info('gtk+-2.0')
        dict_append(plat_info, **gtk_info)
        #x11_info = get_info('x11',notfound_action=1)
        #dict_append(plat_info,**x11_info)

    elif plat == 'gl':
        if sys.platform == 'darwin':
            dict_append(plat_info, **darwin_opengl_opts)
        else:
            msg = "OpenGL build support only on MacOSX right now."
            raise NotImplementedError(msg)


    config.add_extension('_plat_support',
                         sources,
                         **plat_info
                         )

    config.add_data_dir('tests')
    config.add_data_files('*.txt', '*.bat')

    return config
