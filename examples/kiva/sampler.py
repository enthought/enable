#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# Some parts copyright Space Telescope Science Institute.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
# 
# Author: Eric Jones, Enthought Inc.
#------------------------------------------------------------------------------
"""
Set of visual tests exercising the basic drawing styles.  These should work on 
every backend.
"""

import time

import numpy

from enthought.kiva import constants, Font

# This uses the best-fit backend based on the KIVA_WISHLIST environment variable
# or the default list in kiva's top-leve __init__.  To test a specific backend,
# import Canvas and CanvasWindow from backend_NNN.
from enthought.kiva.backend_wx import Canvas, CanvasWindow


#-----------------------------------------------------------------------------
# Utilities constants and functions
#-----------------------------------------------------------------------------

cap_names = {constants.CAP_ROUND:  'round',
             constants.CAP_BUTT:   'butt',
             constants.CAP_SQUARE: 'square'}

join_names = {constants.JOIN_MITER: 'miter',
              constants.JOIN_BEVEL: 'bevel', 
              constants.JOIN_ROUND: 'round'}

def outer_join(a,b):
    return [(aa, bb) for aa in a for bb in b]

def calc_star(size=40):
    half_size = size * .5
    tenth_size = size * .1
    star_pts = [ numpy.array((tenth_size,0)),
                 numpy.array((half_size,size - tenth_size)),
                 numpy.array((size - tenth_size, 0)),
                 numpy.array((0,half_size)),
                 numpy.array((size,half_size)),
               ]
    return numpy.array(star_pts)


class CapSampler:
    def size(self):
        # This really needs to be calculated, but we'll hard code for now.
        return 170,400
                
    def draw(self,gc):
        "Draw a bunch of lines with different line cap settings."
        gc.save_state()    
        gc.set_stroke_color((0,0,0,1))
    
        widths = [1, 3, 5, 10, 20]
        caps   = [constants.CAP_ROUND, constants.CAP_BUTT, constants.CAP_SQUARE]
        gc.translate_ctm(20,20)
        line_length = 120
    
        gc.save_state()
        for cap in caps:
            gc.set_line_cap(cap)
            for width in widths:    
                gc.set_line_width(width)
                gc.translate_ctm(0,10+width)
                gc.begin_path()
                gc.move_to(0,0)
                gc.line_to(line_length,0)
                gc.stroke_path()            
                # Need a little more sophisticated text placement tools.
                gc.set_font_size(6)
                gc.set_text_position(line_length+13,-3)
                gc.show_text("%s" % width)
                
            gc.set_font_size(10)
            gc.set_text_position(5,14)
            gc.show_text("%s" % cap_names[cap])
            gc.translate_ctm(0,20)

        gc.set_font_size(12)
        gc.set_text_position(-10,20)
        gc.show_text("end cap example")

        # restore state to its original form when it was passed in.
        gc.restore_state()
        
        # draw two vertical lines indicating the ends of our lines.
        sz = self.size()
        gc.begin_path()
        gc.move_to(0          ,     0)
        gc.line_to(0          , sz[1]-60)
        gc.move_to(line_length,     0)
        gc.line_to(line_length, sz[1]-60)
        gc.stroke_path()

        gc.restore_state()

    
class DashSampler:

    def __init__(self,width=1,separation=10,dash_values=None,
                 line_cap = constants.CAP_ROUND):
        self.width = width
        self.separation = separation
        self.line_cap = line_cap
        if not dash_values:
            self.dash_values = [0,1,2,3,4,5]
        
    def size(self):
        # This really needs to be calculated, but we'll hard code for now.
        return 170,420

    def draw(self,gc):
        dash_patterns = outer_join(self.dash_values,self.dash_values)
        line_length = 120
        gc.save_state()
        gc.set_line_cap(self.line_cap)
        gc.translate_ctm(10,20)
        gc.set_stroke_color((0,0,0,1))
        gc.set_line_width(self.width)
        
        for dash in dash_patterns:
            gc.set_line_dash(dash)        
            gc.begin_path()
            gc.move_to(0,  0)
            gc.line_to(line_length,0)
            gc.stroke_path()

            #draw dash size next to line
            gc.set_font_size(6)
            gc.set_text_position(line_length+7,-3)
            gc.show_text(str(dash)[1:-1])
            
            gc.translate_ctm(0,self.separation)

        gc.set_font_size(12)
        gc.set_text_position(-5,10)
        gc.show_text("dash %s w=%d" % (cap_names[self.line_cap],
                                                   self.width))
        gc.restore_state()


class JoinSampler:

    def __init__(self,width=1,separation=10,dash_values=None,
                 line_cap = constants.CAP_ROUND):
        self.width = width
        self.separation = separation
        self.line_cap = line_cap
        if not dash_values:
            self.dash_values = [0,1,2,3,4,5]
        
    def size(self):
        # This really needs to be calculated, but we'll hard code for now.
        return 170,250

    def draw(self,gc):        
        width = 10
        gc.save_state()
    
        gc.set_line_width(width)
        gc.set_line_cap(constants.CAP_BUTT)
        
        joins = [constants.JOIN_MITER, constants.JOIN_BEVEL, 
                 constants.JOIN_ROUND]

        gc.translate_ctm(20,40)                 
        for join in joins:
            # we don't want the wedges filled --> alpha=0.
            # (we could use stroke_path instead draw_path)
            gc.set_fill_color((0,0,0,0))
            gc.set_line_join(join)
            gc.begin_path()
            gc.move_to(0,  0)
            gc.line_to(30, 0)
            gc.line_to(30 , 30)
            gc.stroke_path()
            
            #draw dash size next to line
            gc.set_fill_color((0,0,0,1)) # fill color defines text color.
            gc.set_font_size(8)
            gc.set_text_position(5,40)
            gc.show_text(join_names[join])
            
            gc.translate_ctm(40,0)
        
        gc.restore_state()
        
        gc.save_state()
        gc.set_font_size(12)
        gc.set_text_position(20,150)
        gc.show_text("Line joins")
        gc.set_text_position(20,130)
        gc.show_text("width=%d" % width)
        #added to show text extents
        #gc.set_text_position(20,110)
        #text_w,text_h = gc.get_text_extent("foofoo")
        #gc.show_text("text extents" + " " + str(text_w) + " " + str(text_h))
        #end of addition for text extents
        gc.restore_state()


class StarSampler:

    def __init__(self,star_count=20,scale=1.0,rotate=0.0,line_width=3):
        self.star_count = star_count
        symbol_pts = calc_star()
        # connect the last point to the first on the star.
        # PDF doesn't draw it correctly if I do this.
        # now we call close_path in the add_symbol funciton
        #self.symbol_pts = concatenate((symbol_pts,symbol_pts[:1,:]))
        self.symbol_pts = symbol_pts
        numpy.random.seed(10000)
        self.star_pos = numpy.random.randint(20,100,(star_count,2))
        self.scale = scale
        self.rotate = rotate
        self.line_width = line_width

    def add_symbol_slow(self, gc):
        # This is used to be slower by about 1/3than add_star.
        # TODO: verify
        pts = self.symbol_pts
        gc.begin_path()
        gc.move_to(pts[0][0],pts[0][1])
        for i in range(5):
            gc.line_to(pts[i][0],pts[i][1])
            # could do this with a close path...
            gc.close_path()
            gc.draw_path(constants.FILL_STROKE)
    
    def add_symbol(self, gc):
        # add the star to the path
        pts = self.symbol_pts
        gc.lines(pts)
        gc.close_path()

    def size(self):
        # This really needs to be calculated, but we'll hard code for now.
        return 170*self.scale,250*self.scale
    
    def scale(self,scale):
        self.scale = scale
            
    def draw(self,gc):
        gc.save_state()
        # The rotation isn't handled correctly on most platforms.
        
        # fill stars with red
        gc.set_fill_color((1.0,0,0,1))
        
        # outline stars with a wide black line
        gc.set_stroke_color((0,0,0,1))
        gc.set_line_width(self.line_width)

        gc.begin_path()
        for x,y in self.star_pos:
            gc.save_state()           
            gc.translate_ctm(x,y)
            gc.rotate_ctm(self.rotate)
            gc.scale_ctm(self.scale,self.scale)
            self.add_symbol(gc) 
            gc.restore_state()
        gc.draw_path(constants.FILL_STROKE)
        
        gc.set_fill_color((0,0,0,1))
        gc.set_font_size(12)
        gc.set_text_position(5,180)
        gc.show_text("Stars, count=%d" % self.star_count)
        gc.set_text_position(5,160)
        gc.show_text("scale=%2.1f" %  self.scale)
        
        gc.restore_state()


class NoiseSampler:

    def __init__(self, points=20, scale=1.0):
        self.point_count = points
        x = numpy.arange(0.,100.,100./self.point_count)
        y = numpy.random.randint(0,100,(len(x),))
        self.points = numpy.concatenate((x[:,numpy.newaxis],
                                         y[:,numpy.newaxis]), -1)
        self.scale = scale
        
    def size(self):
        # This really needs to be calculated, but we'll hard code for now.
        return 170*self.scale,250*self.scale
        
    def draw(self,gc):
        gc.save_state()
        gc.translate_ctm(20,20)
        gc.scale_ctm(self.scale,self.scale)
        # transparent fill
        gc.set_fill_color((0,0,0,0))
        
        # draw green line.
        gc.set_stroke_color((0,1,0,1))
        gc.set_line_width(1)
        
        gc.begin_path()
        gc.lines(self.points)        
        gc.draw_path(constants.STROKE)
        
        #black border
        # hmm. This doesn't fit around the "plot" on wxPython
        # like I expected it to.
        gc.set_stroke_color((0,0,0,1))
        gc.begin_path()
        gc.rect(0,0,100,100)
        gc.draw_path(constants.STROKE)
        
        gc.set_fill_color((0,0,0,1))
        gc.set_font_size(12)
        gc.set_text_position(5,180)
        gc.show_text("Noise Plot")
        gc.set_text_position(5,160)
        gc.show_text("points=%d" %  self.point_count)

        gc.restore_state()
    
class PolygonSampler:

    def __init__(self):
        self.symbol_pts = calc_star()    
    def size(self):
        # This really needs to be calculated, but we'll hard code for now.
        return 170,250

    def draw(self,gc):
        gc.save_state()
        gc.translate_ctm(10,20)
        # green background
        # black outline
        gc.set_stroke_color((0,0,0,1))
        gc.set_fill_color((0,1,0,1))
        gc.set_line_width(3)
        
        draw_modes = [constants.FILL, constants.EOF_FILL, constants.STROKE, 
                      constants.FILL_STROKE, constants.EOF_FILL_STROKE]
        
        gc.save_state()        
        for mode in draw_modes:
            gc.begin_path()
            gc.lines(self.symbol_pts)
            gc.draw_path(mode)
            gc.translate_ctm(0,40)
        gc.restore_state()
        
        gc.translate_ctm(60,0)                    

        for mode in draw_modes:
            gc.begin_path()
            gc.lines(self.symbol_pts)
            gc.close_path()
            gc.draw_path(mode)
            gc.translate_ctm(0,40)

        gc.restore_state()


class SamplerCanvas(Canvas):
    def __init__(self,parent,samples=()):
        Canvas.__init__(self,parent)
        try:
            # Mac happens to bind the draw method, so we need a fix.
            # figure out how to get this out of the "generic" code.
            self.bind('draw',self._draw)
        except:
            pass    
        self.set_samplers(samples)
        self.font = Font(face_name="Arial")
        return
        
    def set_samplers(self,samples):
        """ Samples "grid" of sampler objects defined by a list of lists
           (2D array) of sampler objects that are laid out on the canvas.
           The first object is drawn in the lower left hand corner based
           on PDF coordinates.
        """
        self.samples = samples
        return
       
    def do_draw(self, gc, vis_rgn=None):
        gc.set_font(self.font)
        gc.save_state()
        w,h = self.size()
        # This approximates a .5 inch or so border so that printers
        # that need the border will still print the entire image.
        gc.translate_ctm(.06*w,.06*h)        
        for row in self.samples:
            max_sy = 0
            gc.save_state()
            for sample in row:            
                t1 = time.clock()            
                gc.save_state()
                sample.draw(gc)
                gc.restore_state()
                t2 = time.clock()
                
                # stamp the example with a run-time
                gc.save_state()
                gc.set_text_position(60,5)
                gc.set_font_size(8)
                gc.show_text("draw time: %3.2f" % (t2-t1))
                
                # draw border around sample
                sx,sy = sample.size()
                gc.begin_path()                
                gc.rect(0,0,sx,sy)
                gc.stroke_path()

                gc.restore_state()                                
                gc.translate_ctm(sx,0)
                max_sy = max(max_sy,sy)        
            
            gc.restore_state()                
            gc.translate_ctm(0,max_sy)

        gc.restore_state()
        gc.flush()
        gc.synchronize()
        return

class SamplerWindow(CanvasWindow):
    """
    A generic window for viewing visual samples.
    """
    def __init__(self, id=-1, title='Sampler Canvas',samples=(),
                 size=(600,800)):
        CanvasWindow.__init__(self,id,title,size,
                                     canvas_class=SamplerCanvas)
        self.canvas.set_samplers(samples)
        return
        
    def set_samplers(self,samples):
        self.canvas.set_samplers(samples)
        return

#-----------------------------------------------------------------------------
# All the above Sampler classes will draw a single example.  The functions
# below group these samples together for display in a window or on a page. 
#
# This is the size of a PDF image on a 8.5x11 sheet of paper.
# We'll use it so that the test samples should fit on a sampler.
#-----------------------------------------------------------------------------

def dash_group(width=1):
    row1 = []
    caps = [constants.CAP_ROUND,constants.CAP_BUTT,constants.CAP_SQUARE]
    for cap in caps:
        row1.append(DashSampler(line_cap=cap,width=width))
    return [row1]

def cap_join_star_noise_group():
    row1 = [CapSampler()]
    row1.append(JoinSampler())
    row1.append(StarSampler(star_count=20))
    row2 = [StarSampler(star_count=300)]
    row2.append(NoiseSampler(points=3000))
    # This one takes forever on PDF for Acrobat to load.
    #row2.append(NoiseSampler(points=30000))
    return [row1,row2]

def single_star_group():
    row1 = [StarSampler(star_count=1,scale=1.0,line_width=1.0)]
    return [row1]

def scaled_star_group():
    row1 = [StarSampler(star_count=1,scale=1.0),
            StarSampler(star_count=1,scale=2.0)]
    return [row1]

def rotated_star_group():
    row1 = [StarSampler(star_count=20),
            StarSampler(star_count=20,rotate=3.14159/4.)]
    return [row1]

def polygon_fill_group():
    row1 = [PolygonSampler()]
    return [row1]

def clipping_group():
    row1 = [ClippingSampler()]
    return [row1]


default_size = (612,792)

all_samples = []

all_samples.append(('Narrow_Dash_Example', dash_group(width=1)))
all_samples.append(('Wide_Dash_Example', dash_group(width=5)))
all_samples.append(('Cap_Join_Star_Noise_Example',
                    cap_join_star_noise_group()))
all_samples.append(('Scaled_Example', scaled_star_group()))
all_samples.append(('Rotated_Example', rotated_star_group()))
all_samples.append(('Polygon_Example', polygon_fill_group()))

all_samples.append(('Single_Thin_Line_Example', single_star_group()))
#all_samples.append(('Clipping_Example', clipping_group()))

def show_all_samplers(default_size=default_size):
    for title,samples in all_samples:
        w = SamplerWindow(title=title,size=default_size)
        w.set_samplers(samples)
        w.Show(1)
    return

if __name__ == "__main__":
    # Create a basic WX app to show a stand-alone window
    try:
        import wx
    except ImportError:
        print "Unable to import WX to display stand-alone test."
        print "Please call sampler.show_all_samplers() from within your application."
    
    class SamplerApp(wx.App):
        def OnInit(self):
            for title,samples in all_samples:
                w = SamplerWindow(title=title,size=default_size)
                w.set_samplers(samples)
                w.Show(1)
                #break
            return 1
    
    app = SamplerApp(False)  # redirect stdout/stderr
    app.MainLoop()
