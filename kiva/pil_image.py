from copy import deepcopy
from io import BytesIO
import os

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from kiva import affine
from kiva.oldagg import CompiledPath
from kiva.basecore2d import GraphicsContextBase
from kiva.constants import EOF_FILL, EOF_FILL_STROKE, FILL, FILL_STROKE, JOIN_ROUND, STROKE, TEXT_FILL, TEXT_FILL_STROKE, TEXT_FILL_STROKE_CLIP, TEXT_STROKE, WEIGHT_BOLD


class GraphicsContext(GraphicsContextBase):

    def __init__(self, size, base_pixel_scale=1.0, *args, **kwargs):
        self.size = size
        self.base_scale = base_pixel_scale
        self.image = Image.new('RGBA', self.size)
        self.bmp_array = np.array(self.image)
        self.image_draw = ImageDraw.Draw(self.image)
        self._scale_factor = None
        self._clip_mask = None
        super().__init__(*args, **kwargs)

    # AbstractGraphicsContext protocol

    def show_text_at_point(self, text, x, y):
        """
        """
        self.show_text(text, (x, y))
    
    def save(self, filename, file_format=None, pil_options={}):

        if file_format is None:
            file_format = ''
        if pil_options is None:
            pil_options = {}

        ext = (
            os.path.splitext(filename)[1][1:] if isinstance(filename, str)
            else ''
        )

        # Check the output format to see if it can handle DPI
        dpi_formats = ('jpg', 'png', 'tiff', 'jpeg')
        if ext in dpi_formats or file_format.lower() in dpi_formats:
            # Assume 72dpi is 1x
            dpi = int(72 * self.base_scale)
            pil_options['dpi'] = (dpi, dpi)

        self.image.save(filename, file_format, **pil_options)

    def to_image(self):
        """ Return the contents of the context as a PIL Image.

        If the graphics context is in BGRA format, it will convert it to
        RGBA for the image.

        Returns
        -------
        img : Image
            A PIL/Pillow Image object with the data in RGBA format.
        """
        return self.image.copy()

    def width(self):
        return self.size[0]
    
    def height(self):
        return self.size[1]

    def clear(self, clear_color=(1.0, 1.0, 1.0, 1.0)):
        pil_color = tuple(int(x * 255) for x in clear_color)
        self.image.paste(pil_color, (0, 0,) + self.size)

    # GrpahicsContext device methods

    def device_prepare_device_ctm(self):
        self.device_ctm = self._coordinate_transform()

    def device_transform_device_ctm(self, func, args):
        self._scale_factor = None
        super().device_transform_device_ctm(func, args)

    def device_set_clipping_path(self, x, y, w, h):
        image = Image.new("L", self.size, color=0)
        draw = ImageDraw.Draw(image)
        transform = self._coordinate_transform()

        x1, y1 = affine.transform_point(transform, (x, y))
        x2, y2 = affine.transform_point(transform, (x + w, y + h))
        draw.rectangle(
            (x1, y1, x2, y2),
            fill=255,
        )
        self._clip_mask = image

    def device_destroy_clipping_path(self):
        self._clip_mask = None

    def device_show_text(self, text):
        """ Draws text on the device at the current text position.

        Advances the current text position to the end of the text.
        """
        # set up the font and aesthetics
        font = self.state.font
        spec = font.findfont()
        scale_factor = self._get_scale_factor()
        font_size = int(font.size * scale_factor)
        pil_font = ImageFont.FreeTypeFont(spec.filename, font_size, spec.face_index)

        if self.state.text_drawing_mode in {TEXT_FILL, TEXT_FILL_STROKE}:
            fill_color = self.state.fill_color.copy()
            fill_color[3] *= self.state.alpha
            fill = tuple((255 * fill_color).astype(int))
        else:
            fill = (0, 0, 0, 0)
        if self.state.text_drawing_mode in {TEXT_STROKE, TEXT_FILL_STROKE}:
            stroke_color = self.state.line_color.copy()
            stroke_color[3] *= self.state.alpha
            stroke = tuple((255 * stroke_color).astype(int))
            stroke_width = self.base_scale
        else:
            stroke = (0, 0, 0, 0)
            stroke_width = 0

        # create an image containing the text
        ascent, descent = pil_font.getmetrics()
        w, h = self.image_draw.textsize(text, pil_font)
        h = max(h, ascent + descent)

        temp_image = Image.new('RGBA', (w + 2*stroke_width, h + 2*stroke_width))
        draw = ImageDraw.Draw(temp_image)
        draw.text((0, 0), text, fill, pil_font, stroke_width=stroke_width, stroke_fill=stroke)
        
        # paste the text into the image
        temp_image = temp_image.transpose(Image.FLIP_TOP_BOTTOM)
        transform = affine.concat(
            self.device_ctm,
            affine.concat(
                self.state.ctm,
                affine.translate(
                    affine.scale(
                        self.state.text_matrix,
                        1/scale_factor, 1/scale_factor,
                    ),
                    0, -descent,
                )
            )
        )
        a, b, c, d, tx, ty = affine.affine_params(affine.invert(transform))
        temp_image = temp_image.transform(
            self.image.size,
            Image.AFFINE,
            (a, b, tx, c, d, ty),
            Image.BILINEAR,
            fillcolor=(0, 0, 0, 0),
        )
        self._compose(temp_image)

        tx, ty = self.get_text_position()
        self.set_text_position(tx + w, ty)

    def device_get_text_extent(self, text):
        font = self.state.font
        spec = font.findfont()
        pil_font = ImageFont.FreeTypeFont(spec.filename, font.size, spec.face_index)
        w, h = self.image_draw.textsize(text, pil_font)
        return (0, font.size - h, w, h)

    def device_update_line_state(self):
        # currently unused - ImageDraw has no public line state
        pass

    def device_update_fill_state(self):
        # currently unused - ImageDraw has no public fill state
        pass

    def device_fill_points(self, pts, mode):
        pts = affine.transform_points(self.device_ctm, pts).reshape(-1).tolist()
        if mode in {FILL, EOF_FILL, FILL_STROKE, EOF_FILL_STROKE}:
            temp_image = Image.new("RGBA", self.size)
            draw = ImageDraw.Draw(temp_image)
            draw.polygon(
                pts,
                fill=tuple((255 * self.state.fill_color).astype(int)),
            )
            self._compose(temp_image)

    def device_stroke_points(self, pts, mode):
        pts = affine.transform_points(self.device_ctm, pts).reshape(-1).tolist()
        scale_factor = self._get_scale_factor()
        if mode in {STROKE, FILL_STROKE, EOF_FILL_STROKE}:
            temp_image = Image.new("RGBA", self.size)
            draw = ImageDraw.Draw(temp_image)
            draw.line(
                pts,
                fill=tuple((255 * self.state.line_color).astype(int)),
                width=int(self.state.line_width * scale_factor),
                joint='curve',
            )
            self._compose(temp_image)

    def device_draw_image(self, image, rect=None):
        if isinstance(image, GraphicsContext):
            image = image.image
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif hasattr(image, "bmp_array"):
            image = Image.fromarray(image.bmp_array)
        if rect is None:
            rect = (0, 0, image.width, image.height)
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        transform = affine.scale(
            affine.translate(
                self.device_ctm,
                rect[0], rect[1],
            ),
            rect[2]/image.width, rect[3]/image.height
        )
        a, b, c, d, tx, ty = affine.affine_params(affine.invert(transform))
        temp_image = image.transform(
            self.image.size,
            Image.AFFINE,
            (a, b, tx, c, d, ty),
            Image.BILINEAR,
            fillcolor=(0, 0, 0, 0),
        )
        self._compose(temp_image)

    # IPython hooks
    
    def _repr_png_(self):
        """ Return a the current contents of the context as PNG image.

        This provides Jupyter and IPython compatibility, so that the graphics
        context can be displayed in the Jupyter Notebook or the IPython Qt
        console.

        Returns
        -------
        data : bytes
            The contents of the context as PNG-format bytes.
        """
        data = BytesIO()
        dpi = 72 * self.base_scale
        self.image.save(data, format='png', dpi=(dpi, dpi))
        return data.getvalue()

    # Private methods

    def _compose(self, image):
        """ Compose a drawing image with the main image with clipping."""
        if self._clip_mask is not None:
            alpha = ImageChops.multiply(image.getchannel("A"), self._clip_mask)
            image.putalpha(alpha)
        self.image.alpha_composite(image)

    def _get_scale_factor(self):
        """A scale factor for the current affine transform.

        This is a suitable factor to use to scale font sizes, line widths,
        etc. or otherwise to get an idea of how
        
        This is the maximum amount a distance will be stretched by the
        linear part of the transform.  It is effectively the operator norm
        of the linear part of the affine transform which can be computed as
        the maximum singular value of the matrix.
        """
        if self._scale_factor is None:
            _, singular_values, _ = np.linalg.svd(self.device_ctm[:2, :2])
            # numpy's svd function returns the singular values sorted
            self._scale_factor = singular_values[0]
        return self._scale_factor

    def _coordinate_transform(self):
        return affine.translate(
            affine.scale(
                affine.affine_identity(),
                self.base_scale,
                -self.base_scale,
            ),
            0.0,
            -self.size[1]/self.base_scale,
        )


def font_metrics_provider():
    """ Creates an object to be used for querying font metrics.
    """
    return GraphicsContext((1, 1))


__all__ = [GraphicsContext, CompiledPath, font_metrics_provider]
