from kiva import agg

gc = agg.GraphicsContextArray((500, 500))
gc.clear()
gc.rect(100, 100, 300, 300)
gc.draw_path()
gc.save("rect.bmp")
