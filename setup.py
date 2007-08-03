from setuptools import setup, find_packages

setup(
    name = 'enthought.enable2',
    version = '3.0a1',
    description  = 'Kiva-based GUI Window and Component package',
    author       = 'Enthought, Inc',
    author_email = 'info@enthought.com',
    url          = 'http://code.enthought.com/ets',
    license      = 'BSD',
    zip_safe     = False,
    packages = find_packages(),
    include_package_data = True,
    install_requires = [
        "enthought.etsconfig",
        "enthought.kiva",
        "enthought.traits",
    ],
    extras_require = {
        "wx": ["enthought.traits.ui.wx"],
        # All non-ets dependencies should be in this extra to ensure users can
        # decide whether to require them or not.
        'nonets': [
            'numpy >=1.0.2',
            ],
        },
    namespace_packages = [
        "enthought",
    ],
)

