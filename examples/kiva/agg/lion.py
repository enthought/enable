import time
import sys

from kiva import agg


if sys.platform == 'win32':
    now = time.clock
else:
    now = time.time

from lion_data import get_lion

def main():
    sz = (1000,1000)

    t1 = now()
    path_and_color, size, center = get_lion()
    t2 = now()
    print(t2 - t1)

    gc = agg.GraphicsContextArray(sz)
    t1 = now()

    gc.translate_ctm(sz[0]/2.,sz[1]/2.)
    Nimages = 90
    for i in range(Nimages):
        for path,color in path_and_color:
            gc.begin_path()
            gc.add_path(path)
            gc.set_fill_color(color)
            gc.set_alpha(0.3)
            gc.fill_path()
        gc.rotate_ctm(1)
    t2 = now()
    print('total time, sec/image, img/sec:', t2 - t1, (t2-t1)/Nimages, Nimages/(t2-t1))
    gc.save('lion.bmp')

if __name__ == "__main__":
    main()

# EOF
