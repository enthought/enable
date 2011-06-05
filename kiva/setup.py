#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#
# Author: Enthought, Inc.
# Description: <Enthought kiva package project>
#------------------------------------------------------------------------------

import sys
import os


def configuration(parent_package=None, top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('kiva', parent_package, top_path)
    config.add_data_dir('tests')
    config.add_data_files('*.txt')

    config.add_subpackage('agg')
    config.add_subpackage('fonttools')
    config.add_subpackage('fonttools.*')
    config.add_subpackage('fonttools.*.*')
    config.add_subpackage('fonttools.*.*.*')
    config.add_data_files('fonttools/fontTools/*.txt')
    config.add_subpackage('trait_defs')
    config.add_subpackage('trait_defs.ui')
    config.add_subpackage('trait_defs.ui.*')

    if sys.platform == 'darwin':
        config.add_subpackage('quartz')

    config.get_version()

    return config
