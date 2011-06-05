from kiva.image import GraphicsContext

gc = GraphicsContext((500,500))
gc.set_fill_color( (1, 0, 0) )
gc.rect(100,100,300,300)
gc.draw_path()
gc.save("simple2_pre.bmp")

# directly manipulate the underlying Numeric array.
# The color tuple is expressed as BGRA.
gc.bmp_array[:100,:100] = (139, 60, 71, 255)
gc.save("simple2_post.bmp")
