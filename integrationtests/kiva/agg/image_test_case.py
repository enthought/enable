import time
import os, sys
import unittest

from PIL import Image
from numpy import alltrue, array, concatenate, dtype, fromstring, newaxis, \
                  pi, ravel, ones, zeros

from kiva import agg
from kiva.fonttools import Font


# alpha blending is approximate in agg, so we allow some "slop" between
# desired and actual results, allow channel differences of to 2.
slop_allowed = 2

UInt8 = dtype('uint8')
Int32 = dtype('int32')

def save(img,file_name):
    """ This only saves the rgb channels of the image
    """
    format = img.format()
    if format == "bgra32":
        size = img.bmp_array.shape[1],img.bmp_array.shape[0]
        bgr = img.bmp_array[:,:,:3]
        rgb = bgr[:,:,::-1].copy()
        st = rgb.tostring()
        pil_img = Image.fromstring("RGB",size,st)
        pil_img.save(file_name)
    else:
        raise NotImplementedError(
            "currently only supports writing out bgra32 images")

def test_name():
    f = sys._getframe()
    class_name = f.f_back.f_locals['self'].__class__.__name__.replace('test_','')
    method_name = f.f_back.f_code.co_name.replace('check_','')
    # !! make sure we delete f or the their are memory leaks that
    # !! result (gc's passed to save are not released)
    del f
    return '.'.join((class_name,method_name))

def sun(interpolation_scheme="simple"):
    pil_img = Image.open('doubleprom_soho_full.jpg')
    img = fromstring(pil_img.tostring(),UInt8)
    img.resize((pil_img.size[1],pil_img.size[0],3))

    alpha = ones(pil_img.size,UInt8) * 255
    img = concatenate((img[:,:,::-1],alpha[:,:,newaxis]),-1).copy()
    return agg.GraphicsContextArray(img,"bgra32", interpolation_scheme)

def solid_bgra32(size,value = 0.0, alpha=1.0):
    img_array = zeros((size[1],size[0],4),UInt8)
    img_array[:,:,:-1] = array(value * 255,UInt8)
    img_array[:,:,-1] = array(alpha * 255,UInt8)
    return img_array

def alpha_blend(src1,src2,alpha=1.0,ambient_alpha = 1.0):
    alpha_ary = src2[:,:,3]/255. * alpha * ambient_alpha
    res = src1[:,:,:] * (1-alpha_ary) + src2[:,:,:] * alpha_ary
    # alpha blending preserves the alpha mask channel of the destination (src1)
    res[:,:,-1] = src1[:,:,-1]
    return res.astype(Int32)

def assert_equal(desired,actual):
    """ Only use for small arrays. """
    try:
        assert alltrue (ravel(actual) == ravel(desired))
    except AssertionError:
        size = sum(array(desired.shape))
        if size < 10:
            diff = abs(ravel(actual.astype(Int32)) -
                       ravel(desired.astype(Int32)))
            msg = '\n'
            msg+= 'desired: %s\n' % ravel(desired)
            msg+= 'actual: %s\n' % ravel(actual)
            msg+= 'abs diff: %s\n' % diff
        else:
            msg = "size: %d.  To large to display" % size
        raise AssertionError(msg)

def assert_close(desired,actual,diff_allowed=2):
    """ Only use for small arrays. """
    try:
        # cast up so math doesn't underflow
        diff = abs(ravel(actual.astype(Int32)) - ravel(desired.astype(Int32)))
        assert alltrue(diff <= diff_allowed)
    except AssertionError:
        size = sum(array(desired.shape))
        if size < 10:
            msg = '\n'
            msg+= 'desired: %s\n' % ravel(desired)
            msg+= 'actual: %s\n' % ravel(actual)
            msg+= 'abs diff: %s\n' % diff
        else:
            msg = "size: %d.  To large to display" % size
        raise AssertionError(msg)

#----------------------------------------------------------------------------
# Tests for various alpha blending cases
#
# These include setting an "ambient" alpha value as well as specifying an
# image with an alpha channel.  Because of the special casing, different
# colors take different paths through the code.  As a result, there are a
# classes to check black, white, and gray images.
#----------------------------------------------------------------------------
class test_alpha_black_image(unittest.TestCase):
    size = (1,1)
    color = 0.0
    def test_simple(self):
        gc = agg.GraphicsContextArray(self.size,pix_format = "bgra32")
        desired = solid_bgra32(self.size,self.color)
        img = agg.GraphicsContextArray(desired, pix_format="bgra32")
        gc.draw_image(img)
        actual = gc.bmp_array
        # for alpha == 1, image should be exactly equal.
        assert_equal(desired,actual)

    def test_image_alpha(self):
        alpha = 0.5
        gc = agg.GraphicsContextArray(self.size,pix_format = "bgra32")
        orig = solid_bgra32(self.size,self.color,alpha)
        img = agg.GraphicsContextArray(orig, pix_format="bgra32")
        gc.draw_image(img)
        actual = gc.bmp_array

        gc_background = solid_bgra32(self.size,1.0)
        orig = solid_bgra32(self.size,self.color,alpha)
        desired = alpha_blend(gc_background,orig)

        # also, the alpha channel of the image is not copied into the
        # desination graphics context, so we have to ignore alphas
        assert_close(desired[:,:,:-1],actual[:,:,:-1],diff_allowed=slop_allowed)

    def test_ambient_alpha(self):
        orig = solid_bgra32(self.size,self.color)
        img = agg.GraphicsContextArray(orig, pix_format="bgra32")
        gc = agg.GraphicsContextArray(self.size,pix_format = "bgra32")
        amb_alpha = 0.5
        gc.set_alpha(amb_alpha)
        gc.draw_image(img)
        actual = gc.bmp_array

        gc_background = solid_bgra32(self.size,1.0)
        orig = solid_bgra32(self.size,self.color)
        desired = alpha_blend(gc_background,orig,ambient_alpha=amb_alpha)
        # alpha blending is approximate, allow channel differences of to 2.
        assert_close(desired,actual,diff_allowed=slop_allowed)

    def test_ambient_plus_image_alpha(self):
        amb_alpha = 0.5
        img_alpha = 0.5
        gc = agg.GraphicsContextArray(self.size,pix_format = "bgra32")
        orig = solid_bgra32(self.size,self.color,img_alpha)
        img = agg.GraphicsContextArray(orig, pix_format="bgra32")
        gc.set_alpha(amb_alpha)
        gc.draw_image(img)
        actual = gc.bmp_array

        gc_background = solid_bgra32(self.size,1.0)
        orig = solid_bgra32(self.size,self.color,img_alpha)
        desired = alpha_blend(gc_background,orig,ambient_alpha=amb_alpha)
        # alpha blending is approximate, allow channel differences of to 2.
        assert_close(desired,actual,diff_allowed=slop_allowed)

class test_alpha_white_image(test_alpha_black_image):
    color = 1.0

class test_alpha_gray_image(test_alpha_black_image):
    color = 0.5


#----------------------------------------------------------------------------
# Test scaling based on "rect" argument to draw_image function.
#
#
#----------------------------------------------------------------------------
class test_rect_scaling_image(unittest.TestCase):
    color = 0.0

    def test_rect_scale(self):
        orig_sz= (10,10)
        img_ary = solid_bgra32(orig_sz,self.color)
        orig = agg.GraphicsContextArray(img_ary, pix_format="bgra32")

        sx,sy = 5,20
        scaled_rect=(0,0,sx,sy)
        gc = agg.GraphicsContextArray((20,20),pix_format = "bgra32")
        gc.draw_image(orig,scaled_rect)
        actual = gc.bmp_array
        save(gc,test_name()+'.bmp')

        desired_sz= (sx,sy)
        img_ary = solid_bgra32(desired_sz,self.color)
        img = agg.GraphicsContextArray(img_ary, pix_format="bgra32")
        gc = agg.GraphicsContextArray((20,20),pix_format = "bgra32")
        gc.draw_image(img)
        desired = gc.bmp_array
        save(gc,test_name()+'2.bmp')
        assert_equal(desired,actual)

    def test_rect_scale_translate(self):
        orig_sz= (10,10)
        img_ary = solid_bgra32(orig_sz,self.color)
        orig = agg.GraphicsContextArray(img_ary, pix_format="bgra32")

        tx, ty = 5,10
        sx,sy = 5,20
        translate_scale_rect=(tx,ty,sx,sy)
        gc = agg.GraphicsContextArray((40,40),pix_format = "bgra32")
        gc.draw_image(orig,translate_scale_rect)
        actual = gc.bmp_array
        save(gc,test_name()+'.bmp')

        desired_sz= (sx,sy)
        img_ary = solid_bgra32(desired_sz,self.color)
        img = agg.GraphicsContextArray(img_ary, pix_format="bgra32")
        gc = agg.GraphicsContextArray((40,40),pix_format = "bgra32")
        gc.translate_ctm(tx,ty)
        gc.draw_image(img)
        desired = gc.bmp_array
        save(gc,test_name()+'2.bmp')
        assert_equal(desired,actual)

#----------------------------------------------------------------------------
# Tests speed of various interpolation schemes
#
#----------------------------------------------------------------------------

class test_text_image(unittest.TestCase):
    def test_antialias(self):
        gc = agg.GraphicsContextArray((200,50),pix_format = "bgra32")
        gc.set_antialias(1)
        f = Font('modern')
        gc.set_font(f)
        gc.show_text("hello")
        save(gc,test_name()+'.bmp')

    def test_no_antialias(self):
        gc = agg.GraphicsContextArray((200,50),pix_format = "bgra32")
        f = Font('modern')
        gc.set_font(f)
        gc.set_antialias(0)
        gc.show_text("hello")
        save(gc,test_name()+'.bmp')

    def test_rotate(self):
        text = "hello"
        gc = agg.GraphicsContextArray((150,150),pix_format = "bgra32")
        f = Font('modern')
        gc.set_font(f)
        tx,ty,sx,sy = bbox = gc.get_text_extent(text)
        gc.translate_ctm(25,25)
        gc.rotate_ctm(pi/2.)
        gc.translate_ctm(0,-sy)
        #gc.show_text(text)
        gc.set_stroke_color([1,0,0])
        gc.set_fill_color([.5,.5,.5])
        gc.rect(tx,ty,sx,sy)
        gc.stroke_path()
        gc.show_text(text)
        save(gc,test_name()+'.bmp')

class test_sun(unittest.TestCase):
    def generic_sun(self,scheme):
        img = sun(scheme)
        sz = array((img.width(),img.height()))
        scaled_sz = sz * .3
        scaled_rect=(0,0,scaled_sz[0],scaled_sz[1])
        gc = agg.GraphicsContextArray(tuple(scaled_sz),pix_format = "bgra32")
        gc.draw_image(img,scaled_rect)
        return gc

    def test_simple(self):
        gc = self.generic_sun("nearest")
        save(gc,test_name()+'.bmp')

    def test_bilinear(self):
        gc = self.generic_sun("bilinear")
        save(gc,test_name()+'.bmp')

    def test_bicubic(self):
        gc = self.generic_sun("bicubic")
        save(gc,test_name()+'.bmp')

    def test_spline16(self):
        gc = self.generic_sun("spline16")
        save(gc,test_name()+'.bmp')

    def test_spline36(self):
        gc = self.generic_sun("spline36")
        save(gc,test_name()+'.bmp')

    def test_sinc64(self):
        gc = self.generic_sun("sinc64")
        save(gc,test_name()+'.bmp')

    def test_sinc144(self):
        gc = self.generic_sun("sinc144")
        save(gc,test_name()+'.bmp')

    def test_sinc256(self):
        gc = self.generic_sun("sinc256")
        save(gc,test_name()+'.bmp')

    def test_blackman100(self):
        gc = self.generic_sun("blackman100")
        save(gc,test_name()+'.bmp')

    def test_blackman256(self):
        gc = self.generic_sun("blackman256")
        save(gc,test_name()+'.bmp')

#----------------------------------------------------------------------------
# Tests speed of various interpolation schemes
#
#
#----------------------------------------------------------------------------
class test_interpolation_image(unittest.TestCase):
    N = 10
    size = (1000,1000)
    color = 0.0

    def generic_timing(self,scheme,size,iters):
        gc = agg.GraphicsContextArray(size,pix_format = "bgra32")
        desired = solid_bgra32(size,self.color)
        img = agg.GraphicsContextArray(desired, pix_format="bgra32",
                                       interpolation=scheme)
        t1 = time.clock()
        for i in range(iters):
            gc.draw_image(img)
        t2 = time.clock()
        img_per_sec = iters/(t2-t1)
        print("'%s' interpolation -> img per sec: %4.2f" % (scheme, img_per_sec))
        return img_per_sec

    def test_simple_timing(self):
        scheme = "nearest"
        return self.generic_timing(scheme,self.size,self.N)

    def test_bilinear_timing(self):
        scheme = "bilinear"
        iters = self.N//2 # this is slower than simple, so use less iters
        return self.generic_timing(scheme,self.size,iters)

    def test_bicubic_timing(self):
        scheme = "bicubic"
        iters = self.N//2 # this is slower than simple, so use less iters
        return self.generic_timing(scheme,self.size,iters)

    def test_sinc144_timing(self):
        scheme = "sinc144"
        iters = self.N//2 # this is slower than simple, so use less iters
        return self.generic_timing(scheme,self.size,iters)


if __name__ == "__main__":
    unittest.main()
