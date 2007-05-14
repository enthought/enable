from os.path import join
from glob import glob
from scipy_distutils.core      import setup
from scipy_distutils.misc_util import get_subpackages, dict_append, get_path
from scipy_distutils.misc_util import merge_config_dicts, default_config_dict

def configuration(parent_package='',parent_path=None):
    package_name = 'enable'
    local_path = get_path(__name__,parent_path)
    
    config = default_config_dict(package_name, parent_package)

    install_path = join(*config['name'].split('.'))

    config['data_files'].extend([
        (join(install_path,'images'),
         glob(join(local_path,'images','*.gif'))),
        (join(install_path,'images'),
         glob(join(local_path,'images','*.png'))),
        (join(install_path,'images'),
         glob(join(local_path,'images','*.ufo'))),
        (install_path,
         [join(local_path,'images.zip')]),
        (join(install_path,'image'),
         [join(local_path,'image','images.zip')]),
        (join(install_path,'image_title'),
         [join(local_path,'image_title','images.zip')]),
        (join(install_path,'image_frame'),
         [join(local_path,'image_frame','images.zip')]),
        ])

    config_list = [config]
    config_list += get_subpackages(local_path,
                                   parent=config['name'],
                                   parent_path=parent_path,
                                   recursive=1,
                                   )
    config_dict = merge_config_dicts(config_list)
    
    return config_dict

if __name__ == '__main__':
    setup(**configuration(parent_path=''))
