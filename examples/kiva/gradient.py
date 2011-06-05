from numpy import array, pi
from os.path import splitext

from enable.kiva_graphics_context import GraphicsContext
from kiva.fonttools import Font
from kiva import constants


def draw(gc):
    # colors are 5 doubles: offset, red, green, blue, alpha
    starting_color = array([0.0, 1.0, 1.0, 1.0, 1.0])
    ending_color = array([1.0, 0.0, 0.0, 0.0, 1.0])

    gc.clear()

    # diagonal
    with gc:
        gc.rect(50,25,150,100)
        gc.linear_gradient(50,25,150,125,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)

    # vertical top to bottom
    with gc:
        gc.rect(50,150,150,50)
        gc.linear_gradient(0,200, 0,150,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)
    # horizontal left to right
    with gc:
        gc.rect(50,200,150,50)
        gc.linear_gradient(50,0, 150,0,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)

    # vertical bottom to top
    with gc:
        gc.rect(50,275,150,50)
        gc.linear_gradient(0,275, 0,325,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)
    # horizontal right to left
    with gc:
        gc.rect(50,325,150,50)
        gc.linear_gradient(200,0, 100,0,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)

    # radial
    with gc:
        gc.arc(325, 75, 50, 0.0, 2*pi)
        gc.radial_gradient(325, 75, 50, 325, 75,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)

    # radial with focal point in upper left
    with gc:
        gc.arc(325, 200, 50, 0.0, 2*pi)
        gc.radial_gradient(325, 200, 50, 300, 225,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)

    # radial with focal point in bottom right
    with gc:
        gc.arc(325, 325, 50, 0.0, 2*pi)
        gc.radial_gradient(325, 325, 50, 350, 300,
                           array([starting_color, ending_color]),
                           "pad")
        gc.draw_path(constants.FILL)

    return


def main():
    gc = GraphicsContext((500, 500))

    gc.scale_ctm(1.25, 1.25)
    draw(gc)

    gc.save(splitext(__file__)[0]+'.png', file_format='png')


if __name__ == '__main__':
    main()
