import platform


def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('gl', parent_package, top_path)
    is_windows = (platform.system() == 'Windows')
    is_macos = (platform.system() == 'Darwin')

    sources = [
        # agg pieces
        'src/agg/agg_bezier_arc.cpp',
        'src/agg/agg_sqrt_tables.cpp',
        'src/agg/agg_trans_affine.cpp',
        # kiva_gl
        'src/kiva_gl_affine_helpers.cpp',
        'src/kiva_gl_compiled_path.cpp',
        'src/kiva_gl_font_type.cpp',
        'src/kiva_gl_graphics_context_base.cpp',
        'src/kiva_gl_graphics_context.cpp',
        'src/kiva_gl_rect.cpp',
    ]
    include_dirs = ['src', 'src/agg', numpy.get_include()]
    macros = []
    if is_macos:
        macros = [
            ('__DARWIN__', None),
            # OpenGL is deprecated starting with macOS 10.14 and gone in 10.15
            # But that doesn't mean we want to hear about it. We know, Apple.
            ('GL_SILENCE_DEPRECATION', None),
        ]

    config.add_library(
        'kiva_gl_src', sources,
        include_dirs=include_dirs,
        # Use "macros" instead of "define_macros" because the
        # latter is only used for extensions, and not clibs
        macros=macros,
    )

    include_dirs.append('src/swig')

    build_libraries = ['kiva_gl_src']
    extra_link_args = []
    define_macros = []
    if is_windows:
        # Visual studio does not support/need these
        extra_compile_args = []
    else:
        extra_compile_args = [
           '-Wfatal-errors',
           '-Wno-unused-function',
        ]

    if is_windows:
        build_libraries += ['opengl32', 'glu32']
    elif is_macos:
        # Options to make macOS link OpenGL
        if '64bit' not in platform.architecture():
            darwin_frameworks = ['Carbon', 'ApplicationServices', 'OpenGL']
        else:
            darwin_frameworks = ['ApplicationServices', 'OpenGL']

        for framework in darwin_frameworks:
            extra_link_args.extend(['-framework', framework])

        include_dirs.extend([
            '/System/Library/Frameworks/%s.framework/Versions/A/Headers' % x
            for x in darwin_frameworks
        ])
        define_macros = [
            ('__DARWIN__', None),
            # OpenGL is deprecated starting with macOS 10.14 and gone in 10.15
            # But that doesn't mean we want to hear about it. We know, Apple.
            ('GL_SILENCE_DEPRECATION', None),
        ]
    else:
        # This should work for most linuxes (linuces?)
        build_libraries += ['GL', 'GLU']

    config.add_extension(
        '_gl',
        sources=['gl.i'],
        include_dirs=include_dirs,
        define_macros=define_macros,
        libraries=build_libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++',
    )
    return config
