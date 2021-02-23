import unittest

from PIL import Image

from kiva import agg


def save(img, file_name):
    """ This only saves the rgb channels of the image
    """
    format = img.format()
    if format == "bgra32":
        bgr = img.bmp_array[:, :, :3]
        rgb = bgr[:, :, ::-1].copy()
        pil_img = Image.fromarray(rgb, "RGB")
        pil_img.save(file_name)
    else:
        raise NotImplementedError(
            "currently only supports writing out " "bgra32 images"
        )


class TestDrawDash(unittest.TestCase):
    def test_dash(self):
        gc = agg.GraphicsContextArray((100, 100))
        gc.set_line_dash([2, 2])
        for i in range(10):
            gc.move_to(0, 0)
            gc.line_to(0, 100)
            gc.stroke_path()
            gc.translate_ctm(10, 0)
        save(gc, "dash.bmp")
