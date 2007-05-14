#!/usr/bin/env python

import os

def configuration(parent_package='enthought',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('enable2',parent_package,top_path)
    
    #add the parent __init__.py to allow for importing
    config.add_data_files(('..', os.path.abspath(os.path.join('..','__init__.py'))))
    
    
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_data_dir('demo')

    config.add_subpackage('drawing')
    config.add_data_dir('drawing/tests')

    config.add_subpackage('image')
    #config.add_data_dir('image/images')
    config.add_data_files('image/*.zip')

    #config.add_subpackage('image_frame')
    #config.add_data_dir('image_frame/images')
    #config.add_data_files('image_frame/*.zip')

    #config.add_subpackage('image_title')
    #config.add_data_dir('image_title/images')
    #config.add_data_files('image_title/*.zip')

    config.add_data_dir('images')

    config.add_subpackage('primitives')

    config.add_data_dir('tests')
    
    config.add_subpackage('tk')
    config.add_subpackage('wx_backend')
    
    config.add_data_files('*.zip')
    config.add_data_files('*.txt')
    
    return config

if __name__ == "__main__":
    try:
        from numpy.distutils.core import setup
    except ImportError:
        execfile('setup_enable.py')
    else:
        setup(version='2.1.0',
           description  = 'Kiva-based GUI Window and Component package',
           author       = 'Enthought, Inc',
           author_email = 'info@enthought.com',
           url          = 'http://code.enthought.com/chaco',
           license      = 'BSD',
           zip_safe     = False,
           configuration=configuration)
