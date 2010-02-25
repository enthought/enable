# Function to convert simple ETS project names and versions to a requirements
# spec that works for both development builds and stable builds.  Allows
# a caller to specify a max version, which is intended to work along with
# Enthought's standard versioning scheme -- see the following write up:
#    https://svn.enthought.com/enthought/wiki/EnthoughtVersionNumbers
def etsdep(p, min, max=None, literal=False):
    require = '%s >=%s.dev' % (p, min)
    if max is not None:
        if literal is False:
            require = '%s, <%s.a' % (require, max)
        else:
            require = '%s, <%s' % (require, max)
    return require


# Declare our ETS project dependencies.
#CHACO -- not ever imported here, the etscollect tool is messing up somehow.
ENTHOUGHTBASE_UI = etsdep('EnthoughtBase[ui]', '3.0.4')
TRAITSBACKENDWX = etsdep('TraitsBackendWX', '3.3.0')
TRAITSGUI = etsdep('TraitsGUI', '3.3.0')
TRAITS_UI = etsdep('Traits[ui]', '3.3.0')


# A dictionary of the setup data information.
INFO = {
    'extras_require': {
        'ps': [],
        'svg': [],
        'traits': [],

        # All non-ets dependencies should be in this extra to ensure users can
        # decide whether to require them or not.
        'nonets': [
            "numpy >= 1.1.0",
            ],
        },
    'install_requires': [
        ENTHOUGHTBASE_UI,
        TRAITSGUI,
        TRAITS_UI,
        ],
    'name': 'Enable',
    'version': '3.3.0',
    }
