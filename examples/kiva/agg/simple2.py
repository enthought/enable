from kiva import agg

gc = agg.GraphicsContextArray((500, 500))
gc.rect(100, 100, 300, 300)
gc.draw_path()
gc.save("rect.bmp")

# directly manipulate the underlying
# Numeric array.
gc.bmp_array[:100, :100] = 255 * .5
gc.save("rect2.bmp")
