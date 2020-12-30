#!/usr/bin/env python
import sys

from numpy import get_include


def configuration(parent_package=None, top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('kiva', parent_package, top_path)
    config.add_data_dir('tests')
    config.add_data_files('*.txt')

    config.add_subpackage('agg')
    config.add_subpackage('fonttools')
    config.add_subpackage('fonttools.tests')
    config.add_subpackage('trait_defs')
    config.add_subpackage('trait_defs.ui')
    config.add_subpackage('trait_defs.ui.*')

    if sys.platform == 'darwin':
        config.add_subpackage('quartz')

    config.get_version()

    ext_sources = ['_cython_speedups.cpp', '_hit_test.cpp']
    config.add_extension('_cython_speedups', sources=ext_sources,
                         include_dirs=['.', get_include()])

    return config
