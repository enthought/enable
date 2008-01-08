import wx
from numpy import min, sum, where, uint8, int32

from enthought.kiva.backend_wx import Canvas
from enthought.kiva.backend_image import Image
from enthought.kiva import Font
from enthought.traits.api import Enum, HasTraits, Range

import time

def fixpath(path):
    import os
    return os.path.join(os.path.dirname(__file__),path)

class SmartImage(HasTraits):
    x = Range(-50.0, 50.0, 0.0)
    y = Range(-50.0, 50.0, 0.0)
    rotation = Range(0.0, 360.0, 0.0)
    scale = Range(0.1, 3.0, 1.0)
    opacity = Range(0.0, 1.0, 0.6)
    interpolation = Enum("nearest", "bilinear", "bicubic", "spline16")

    def __init__(self,file,fit_to_screen=0):                       
        self.fit_to_screen = fit_to_screen
        self.image = Image(file)

    def draw(self,gc):
        # !! Note: Code will not work if gc ctm is not identity.
        # !!       We should look at fixing this.
        gc.save_state()
        gc_sz = gc.width(), gc.height()
        w = self.image.width()
        h = self.image.height()  

        wr = gc.width()/float(w)
        hr = gc.height()/float(h)
        ratio = min(wr,hr)

        self.image.set_image_interpolation(self.interpolation.lower())
        if self.fit_to_screen:
            if wr == ratio:
                gc.translate_ctm(0, (gc.height() - h*ratio)/4.)
            else:
                gc.translate_ctm((gc.width() - w*ratio)/2.,0)

            gc.draw_image(self.image,(0,0,ratio*w, ratio*h))
        else:            
            scale = self.scale * ratio
            # move to center of screen
            gc.set_alpha(self.opacity)
            # scale image            
            gc.scale_ctm(scale,scale)            
            # move origin to middle of image
            gc.translate_ctm(-w/2,-h/2)            
            
            # translate image to middle of screen
            gc.translate_ctm(gc.width()/scale/2, gc.height()/scale/2)
            # now translate to x,y postions set by traits.
            gc.translate_ctm(self.x, self.y)
            
            gc.rotate_ctm(self.rotation * 3.1416/180.)            
            gc.draw_image(self.image)
        gc.restore_state()


class NewCanvas(Canvas):
                 
    def __init__(self, parent, id = -1, size = wx.DefaultSize):
        Canvas.__init__(self,parent,id,size)
        self.clear_color = (0,0,0)
        self.objects = []
        self.font = Font("Arial")

    def update_window(self):
        self.dirty = 1
        self.Refresh()

    def add(self,obj):
        self.objects.append(obj)

    def do_draw(self,gc):
        gc.save_state()
        t1 = time.clock()
        gc.set_font(self.font)

        for obj in self.objects:
            obj.draw(gc)

        gc.restore_state()
        t2 = time.clock()
        self.image_time = t2-t1

        gc.save_state()
        gc.set_fill_color((1,1,1))
        gc.set_stroke_color((1,1,1))
        self.total_paint_time = (self.image_time +  
                                 self.blit_time + self.clear_time)
        text = "total time: %3.3f" % self.total_paint_time

        sz = gc.width(), gc.height()
        gc.show_text(text,(10,sz[1] - 20))
        text = "frames/sec: %3.3f" % (1.0/self.total_paint_time)
        gc.show_text(text,(10,sz[1] - 40))

        gc.restore_state()
        #print self.image_time, self.blit_time, self.clear_time
                
class BrainCanvasWindow(wx.Frame):
    def __init__(self, id=-1, title='Brain Example',size=(600,800)):
        parent = None
        wx.Frame.__init__(self, parent, id, title, size=size)
        
        canvas = NewCanvas(self)
        canvas.SetSize((500,500))
                
        brain1 = SmartImage(fixpath("brain1.gif"),fit_to_screen=1)
        brain2 = SmartImage(fixpath("brain2.gif"))
        
        # set the opacity of the image (alpha channel)
        rgb = brain2.image.bmp_array[:,:,:2].astype(int32)
        mask = where(sum(rgb,axis=-1) > 0, 255, 0)
        brain2.image.bmp_array[:,:,3] = mask.astype(uint8)
                        
        canvas.add(brain1)
        canvas.add(brain2)
        
        ui = brain2.edit_traits(parent=self, kind='subpanel')
        control_panel = ui.control

        brain2.on_trait_change(canvas.update_window)
                
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)
        self.SetAutoLayout(True)

        sizer.Add(canvas, 1, wx.EXPAND)
        sizer.Add(control_panel, 0, wx.EXPAND)

        sizer.Fit(self)

        self.Show(1)        

def main():
    
    class MyApp(wx.App):
        def OnInit(self):
            BrainCanvasWindow(size=(1000,1000))
            return 1
    
    app = MyApp(0)
    app.MainLoop()

if __name__ == "__main__":    
    main()
