"""
Compatibility module to help support various versions of PIL and
Pillow.

"""
import PIL

PIL_VERSION = getattr(PIL, 'VERSION')
PILLOW_VERSION = getattr(PIL, 'PILLOW_VERSION', PIL_VERSION)


def piltostring(image):
    if PILLOW_VERSION.starts_with('3'):
        return image.tobytes()
    else:
        return image.tostring()
