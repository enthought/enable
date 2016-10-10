

import os
import time
from math import pi

from kiva import agg
from kiva.fonttools import Font

ArialFont = Font('arial')

def save_path(filename):
    return filename

def draw_text(gc, text, bbox, text_color, bbox_color):
    gc.set_stroke_color(bbox_color)
    gc.rect(bbox[0],bbox[1],bbox[2],bbox[3])
    gc.stroke_path()
    gc.set_fill_color(text_color)
    gc.show_text(text)

def main():
    bbox_color = [1.0,0.0,0.0]
    text_color = [0.0,0.0,1.0,.5]
    text = "hello"

    tot1 = time.clock()

    t1 = time.clock()
    gc=agg.GraphicsContextArray((800,800))
    gc.set_font(ArialFont)
    gc.set_alpha(1.0)
    bbox = gc.get_text_extent(text)
    draw_text(gc,text,bbox,text_color,bbox_color)
    t2 = time.clock()
    print('1st:', t2-t1)

    t1 = time.clock()
    with gc:
        gc.translate_ctm(50,50)
        gc.rotate_ctm(pi/4)
        draw_text(gc,text,bbox,text_color,bbox_color)
    t2 = time.clock()
    print('2nd:', t2-t1)

    t1 = time.clock()
    with gc:
        gc.translate_ctm(100,100)
        gc.scale_ctm(4.0,2.0)
        draw_text(gc,text,bbox,text_color,bbox_color)
    t2 = time.clock()
    print('3rd:', t2-t1)

    t1 = time.clock()
    with gc:
        gc.translate_ctm(200,200)
        gc.scale_ctm(4.0,2.0)
        gc.rotate_ctm(pi/4)
        draw_text(gc,text,bbox,text_color,bbox_color)
    t2 = time.clock()
    print('4th:', t2-t1)
    print('tot:', time.clock() - tot1)
    gc.save(save_path('text2.bmp'))

    import random
    import string
    alpha = list(string.ascii_letters) + list('012345679')

    N =100
    strs = []
    for i in range(N):
        random.shuffle(alpha)
        strs.append(''.join(alpha))
    print('starting:')
    t1 = time.clock()
    for s in strs:
        gc.show_text(s)
    t2 = time.clock()
    print()
    print('1. %d different 62 letter strings(total,per string):' % N)
    print('    %f %f' % (t2-t1,((t2-t1)/N)))

    t1 = time.clock()
    for i in range(N/10):
        for s in strs[:10]:
            gc.show_text(s)
    t2 = time.clock()
    print('2. 10 strings with 62 letter rendered %d times (total,per str):' % N)
    print('    %f %f' % (t2-t1,((t2-t1)/N)))

    print("Version 2. above is common in graphs and should be about 10 ")
    print("times faster than the first because of caching")
    print()
    gc.save(save_path('text2.bmp'))

def main2():

    from PIL import Image
    pil_img = Image.open('doubleprom_soho_full.jpg')
    img = fromstring(pil_img.tostring(),UInt8)
    img = img.resize((pil_img.size[1],pil_img.size[0],3))

    alpha = ones(pil_img.size,UInt8) * 255
    img = concatenate((img[:,:,::-1],alpha[:,:,NewAxis]),-1).copy()
    print('typecode:', typecode(img)    , iscontiguous(img))
    print(shape(img))
    gc=agg.GraphicsContextArray((1000,1000))
    gc.draw_image(img)
    print(pil_img.getpixel((300,300)), img[300,300], gc.bmp_array[300,300])
    gc.save('sun.bmp')

def main3():

    from PIL import Image
    pil_img = Image.open('doubleprom_soho_full.jpg')
    img = fromstring(pil_img.tostring(),UInt8)
    img = img.resize((pil_img.size[1],pil_img.size[0],3))

    alpha = ones(pil_img.size,UInt8) * 255
    img = concatenate((img[:,:,::-1],alpha[:,:,NewAxis]),-1).copy()
    print('typecode:', typecode(img)    , iscontiguous(img))
    print(shape(img))
    agg_img = agg.Image(img,"bgra32", interpolation_scheme="simple")
    gc=agg.GraphicsContextArray((760,760))
    N = 100
    gc.show_text("SUN")
    t1 = time.clock()
    for i in range(N):
        with gc:
            #gc.rotate_ctm(.2)
            gc.set_alpha(1.0)
            gc.draw_image(agg_img)
            #print pil_img.getpixel((300,300)), img[300,300], gc.bmp_array[300,300]
            gc.translate_ctm(150,300)
            gc.scale_ctm(10,10)
            gc.set_fill_color((0.0,0,1.0,.25))
            #gc.show_text("SUN")
    t2 = time.clock()
    print("images per second: %g" % (N/(t2-t1)))
    gc.save('sun3.bmp')

def main4():
    """ Test drawing an rgb24 into a bgra32"""
    from PIL import Image
    pil_img = Image.open('doubleprom_soho_full.jpg')
    img = fromstring(pil_img.tostring(),UInt8)
    img = img.resize((pil_img.size[1],pil_img.size[0],3))
    print('typecode:', typecode(img)    , iscontiguous(img))
    print(shape(img))
    agg_img = agg.Image(img,"rgb24", interpolation_scheme="simple")
    gc=agg.GraphicsContextArray((1000,1000))
    N = 1
    t1 = time.clock()
    for i in range(N):
        with gc:
            #gc.rotate_ctm(.2)
            #gc.set_alpha(0.5)
            gc.draw_image(agg_img)
            #print pil_img.getpixel((300,300)), img[300,300], gc.bmp_array[300,300]
            gc.translate_ctm(150,300)
            gc.scale_ctm(10,10)
            gc.set_fill_color((0.0,0,1.0,.5))
            #gc.show_text("SUN")
    t2 = time.clock()
    print("images per second: %g" % (N/(t2-t1)))
    gc.save('sun2.bmp')

def main5(gc):
    bbox_color = [1.0,0.0,0.0]
    text_color = [0.0,0.0,1.0,1.0]
    text = "hello"

    tot1 = time.clock()

    t1 = time.clock()
    gc.set_alpha(1.0)
    bbox = gc.get_text_extent(text)
    draw_text(gc,text,bbox,text_color,bbox_color)
    t2 = time.clock()
    print('1st:', t2-t1)


    import random
    import string
    strs = ['012345679',
            'abcdefghi',
            'jklmnopqr',
            'stuvwxyzA',
            'BCDEFGHIJ',
            'KLMNOPQRS',
            'TUVWXYZ!@',
            '#$%^&*()-',
            '=+{[}]|\:',
            '<,>.?/~`"']

    N =100
    t1 = time.clock()
    for i in range(N):
        for s in strs:
            #gc.translate_ctm(0,14)
            gc.show_text(s)
    t2 = time.clock()
    print()
    print(' %d  strings(total,per string):' % (N*10))
    print('    %f %f' % (t2-t1,((t2-t1)/(N*10))))
    gc.save('text2.bmp')


if __name__ == '__main__':

    import profile
    gc=agg.GraphicsContextArray((800,800))
    #profile.run('main()')
    #main5(gc)
    #main5(gc)
    #main5(gc)
    #profile.run('main5(gc)')
    #main()
    #main2()
    #main3()
    #main4()

    main()
    #main2(gc)
    #main4(gc)
    #main5(gc)
