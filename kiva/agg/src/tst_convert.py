import agg
q=agg.Image((10,10),pix_format="rgb24")
q.convert_pixel_format("rgba32")
print q.format()
