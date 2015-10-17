"""
Compatibility module to help support various versions of PIL and
Pillow.

"""
import PIL
from PIL import Image

PILLOW_VERSION = getattr(PIL, 'PILLOW_VERSION', PIL.VERSION)


def piltostring(image):
    if PILLOW_VERSION.starts_with('3'):
        return image.tobytes()
    else:
        return image.tostring()


def pilfromstring(*args, **kwargs):
    if PILLOW_VERSION.starts_with('3'):
        return Image.frombytes(*args, **kwargs)
    else:
        return Image.tostring(*args, **kwargs)
