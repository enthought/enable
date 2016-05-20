import unittest

from kiva import agg
from kiva.compat import pilfromstring


def save(img, file_name):
    """ This only saves the rgb channels of the image
    """
    format = img.format()
    if format == "bgra32":
        size = (img.bmp_array.shape[1], img.bmp_array.shape[0])
        bgr = img.bmp_array[:, :, :3]
        rgb = bgr[:, :, ::-1].copy()
        st = rgb.tostring()
        pil_img = pilfromstring("RGB", size, st)
        pil_img.save(file_name)
    else:
        raise NotImplementedError("currently only supports writing out "
                                  "bgra32 images")


class TestDrawDash(unittest.TestCase):
    def test_dash(self):
        gc = agg.GraphicsContextArray((100, 100))
        gc.set_line_dash([2, 2])
        for i in range(10):
            gc.move_to(0, 0)
            gc.line_to(0, 100)
            gc.stroke_path()
            gc.translate_ctm(10, 0)
        save(gc, 'dash.bmp')


if __name__ == "__main__":
    unittest.main()
