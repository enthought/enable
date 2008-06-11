#!/usr/bin/env python

import os
import sys

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import dict_append, get_info
    config = Configuration('mac',parent_package,top_path)

    wx_info = get_info('wx')
    # Find the version number of wx.
    release = '2.6'
    for macro, value in wx_info['define_macros']:
        if macro.startswith('WX_RELEASE_'):
            release = macro[len('WX_RELEASE_'):].replace('_', '.')
            break

    def generate_c_from_pyrex(extension, build_dir):
        if not sys.platform == 'darwin':
            print 'No %s will be built for this platform.' % (extension.name)
            return
        from distutils.dep_util import newer_group
        name = extension.name.split('.')[-1]
        source = extension.depends[0]
        target = os.path.join(build_dir, name+'.c')

        if newer_group(extension.depends, target):
            from Pyrex.Compiler import Main
            options = Main.CompilationOptions(
                defaults=Main.default_options,
                output_file=target)
            pyrex_result = Main.compile(source, options=options)
            if pyrex_result.num_errors != 0:
                raise RuntimeError("%d errors in Pyrex compile" % 
                    pyrex_result.num_errors)
        return target


    extra_link_args=[
        '-Wl,-framework', '-Wl,CoreFoundation',
        '-Wl,-framework', '-Wl,ApplicationServices',
        '-Wl,-framework', '-Wl,Carbon',
        '-Wl,-framework', '-Wl,Foundation',
        ]
    include_dirs = ['/Developer/Headers/FlatCarbon']
    config.add_extension('ATSFont',
                         [generate_c_from_pyrex],
                         include_dirs = include_dirs,
                         extra_link_args = extra_link_args,
                         depends=["ATSFont.pyx",
                                  "Python.pxi",
                                  "ATS.pxi",
                                  ],
                         )
    config.add_extension('ABCGI',
                         [generate_c_from_pyrex],
                         include_dirs = include_dirs,
                         depends = ["ABCGI.pyx",
                                    "ATSUI.pxi",
                                    "Python.pxi",
                                    "numpy.pxi",
                                    "c_numpy.pxd",
                                    "CoreFoundation.pxi",
                                    "CoreGraphics.pxi",
                                    "QuickDraw.pxi",
                                    ]
                         )

    if release == '2.6':
        macport_cpp = config.paths('macport26.cpp')[0]
    else:
        macport_cpp = config.paths('macport28.cpp')[0]
    

    def get_macport_cpp(extension, build_dir):
        if sys.platform != 'darwin':
            print 'No %s will be built for this platform.' % (extension.name)
            return None

        elif release not in ('2.6', '2.8'):
            print ('No %s will be built because we do not recognize wx version '
                '%s' % (extension.name, release))
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

