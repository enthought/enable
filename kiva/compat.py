"""
Compatibility module to help support various versions of PIL and
Pillow.

"""
from PIL import Image

HAS_FROM_BYTES = hasattr(Image, 'frombytes')


def piltostring(image):
    if hasattr(image, 'tobytes'):
        return image.tobytes()
    else:
        return image.tostring()


def pilfromstring(*args, **kwargs):
    if HAS_FROM_BYTES:
        return Image.frombytes(*args, **kwargs)
    else:
        return Image.fromstring(*args, **kwargs)
