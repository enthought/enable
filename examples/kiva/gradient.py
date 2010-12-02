from enthought.kiva import GraphicsContext
import numpy

# colors are 5 doubles: offset, red, green, blue, alpha
starting_color = numpy.array([0.0, 1.0, 1.0, 1.0, 1.0])
ending_color = numpy.array([1.0, 0.0, 0.0, 0.0, 1.0])

gc = GraphicsContext((500,500))
gc.clear()
gc.rect(100,100,300,300)
gc.linear_gradient(100, 100, 300, 300,
                    numpy.array([starting_color, ending_color]),
                    "pad", 'objectBoundingBox')
gc.draw_path()
gc.save("gradient.bmp")
