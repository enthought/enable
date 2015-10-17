"""
Compatibility module to help support various versions of PIL and
Pillow.

"""
from PIL import Image

HAS_FROM_STRING = hasattr(Image, 'fromstring')


def piltostring(image):
    if hasattr(image, 'tostring'):
        return image.tostring()
    else:
        return image.tobytes()


def pilfromstring(*args, **kwargs):
    if HAS_FROM_STRING:
        return Image.fromstring(*args, **kwargs)
    else:
        return Image.frombytes(*args, **kwargs)
