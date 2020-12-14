from tvtk.api import tvtk
from mayavi import mlab
from enable.vtk_backend.vtk_window import EnableVTKWindow

def main():
    from basic_move import Box
    from enable.api import Container
    container = Container()
    box = Box(bounds=[30,30], position=[20,20], padding=5)
    container.add(box)

    # Create the mlab test mesh and get references to various parts of the
    # VTK pipeline
    m = mlab.test_mesh()
    scene = mlab.gcf().scene
    render_window = scene.render_window
    renderer = scene.renderer
    rwi = scene.interactor

    # Create the Enable Window
    window = EnableVTKWindow(rwi, renderer,
            component=container,
            #istyle_class = tvtk.InteractorStyleSwitch,
            istyle_class = tvtk.InteractorStyle,
            resizable = "v",
            bounds = [100, 100],
            padding_top = 20,
            padding_bottom = 20,
            padding_left = 20,
            )

    #rwi.render()
    #rwi.start()
    mlab.show()
    return window, render_window

if __name__=="__main__":
    main()
