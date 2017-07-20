#!/usr/bin/env python

import os
import platform
import sys

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import dict_append, get_info
    config = Configuration('quartz', parent_package, top_path)

    def generate_c_from_cython(extension, build_dir):
        if not sys.platform == 'darwin':
            print('No %s will be built for this platform.' % extension.name)
            return
        from distutils.dep_util import newer_group
        name = extension.name.split('.')[-1]
        source = extension.depends[0]
        target = os.path.join(build_dir, name+'.c')

        if newer_group(extension.depends, target):
            from Cython.Compiler import Main
            options = Main.CompilationOptions(
                defaults=Main.default_options,
                output_file=target)
            cython_result = Main.compile(source, options=options)
            if cython_result.num_errors != 0:
                raise RuntimeError("%d errors in Cython compile" %
                    cython_result.num_errors)
        return target

    frameworks = ['Cocoa', 'CoreFoundation','ApplicationServices','Foundation']
    extra_link_args=['-framework %s' % x for x in frameworks]
    include_dirs = [
        '/System/Library/Frameworks/%s.framework/Versions/A/Headers' % x
        for x in frameworks
    ]

    config.add_extension('ABCGI',
                         [generate_c_from_cython],
                         extra_link_args = extra_link_args,
                         depends = ["ABCGI.pyx",
                                    "Python.pxi",
                                    "numpy.pxi",
                                    "c_numpy.pxd",
                                    "CoreFoundation.pxi",
                                    "CoreGraphics.pxi",
                                    "CoreText.pxi",
                                    ]
                         )

    config.add_extension('CTFont',
                         [generate_c_from_cython],
                         extra_link_args = extra_link_args,
                         depends=["CTFont.pyx",
                                  "CoreFoundation.pxi",
                                  "CoreGraphics.pxi",
                                  "CoreText.pxi",
                                  ],
                        )

    config.add_extension("mac_context",
                         ["mac_context.c", "mac_context_cocoa.m"],
                         include_dirs = include_dirs,
                         extra_link_args = extra_link_args,
                         depends = ["mac_context.h"],
                         )

    wx_info = get_info('wx')
    if wx_info:
        wx_release = '2.6'
        for macro, value in wx_info['define_macros']:
            if macro.startswith('WX_RELEASE_'):
                wx_release = macro[len('WX_RELEASE_'):].replace('_', '.')
                break

        # only build macport for wxPython version 2.6, it's not needed in the
        # newer releases (see __init__.py)
        if wx_release == '2.6':
            macport_cpp = config.paths('macport26.cpp')[0]

            def get_macport_cpp(extension, build_dir):
                if sys.platform != 'darwin':
                    print('No %s will be built for this platform.'
                          % extension.name)
                    return None

                elif wx_release not in ('2.6', '2.8'):
                    print ('No %s will be built because we do not recognize '
                           'wx version %s' % (extension.name, wx_release))
                    return None

                return macport_cpp

            info = {}
            dict_append(info, define_macros=[("__WXMAC__", 1)])
            dict_append(info, **wx_info)
            config.add_extension('macport', [get_macport_cpp],
                                 depends = [macport_cpp],
                                 **wx_info
                                 )
    return config
