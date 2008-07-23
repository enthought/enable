import os, wx
from enthought.kiva import Canvas
from enthought.kiva.fonttools import Font, str_to_font
from enthought.kiva.backend_image import Image
from enthought.traits.api import Enum, HasTraits, Range

import time

class BrainModel(HasTraits):
    x = Range(-0.5, 0.5, 0.3855)
    y = Range(-0.5, 0.5, 0.4019)
    rotation = Range(0.0, 360.0, 0.0)
    scale = Range(0.1, 3.0, 2.23)
    opacity = Range(0.0, 1.0, 0.5)
    interpolation = Enum("nearest","bilinear","bicubic",
                                  "spline16","blackman100")

class BrainCanvas(Canvas):
                 
    def __init__(self, parent, id = -1, size = wx.DefaultSize):
        Canvas.__init__(self,parent,id,size)

        dirname = os.path.dirname(os.path.abspath(__file__))
        self.brain1 = Image(os.path.join(dirname, "brain1.gif"),
                            interpolation="nearest")                                           
        self.brain2 = Image(os.path.join(dirname, "brain2.gif"),
                            interpolation="nearest")                                           
        # set the alpha channel to the same as the "blue" channel
        self.brain2.bmp_array[:,:,3] = self.brain2.bmp_array[:,:,0]
        
        self.model = BrainModel()
        self.model.on_trait_change(self.update_window)

    def controls_ui(self, parent=None, kind='modal'):
        return self.model.edit_traits(parent=parent, kind=kind).control

    def update_window(self):
        self.dirty=1
        self.Refresh()
                                                       
    def do_draw(self,gc):
        sz = self.GetClientSizeTuple()
        gc.set_font(str_to_font("modern 10"))
        gc.save_state()
        
        t1 = time.clock()
        self.brain1.set_image_interpolation(self.model.interpolation)
        img_box = (0, 0, sz[0], sz[1])
        gc.draw_image(self.brain1, img_box)
        t2 = time.clock()
        self.image_time = t2 - t1
        
        t1 = time.clock()        
        # default to centered in window
        gc.translate_ctm(sz[0]/2,sz[1]/2)

        # now handle user offsets.
        gc.translate_ctm(self.model.x*sz[0], self.model.y*sz[1])

        # the lion is upside-down to start with -- turn right side up.
        gc.rotate_ctm(3.1416)

        # now add the user rotation setting
        gc.rotate_ctm(self.model.rotation * 3.1416/180.)
        gc.scale_ctm(self.model.scale, self.model.scale)
        gc.set_alpha(self.model.opacity)

        self.brain2.set_image_interpolation(self.model.interpolation)
        gc.draw_image(self.brain2)
        t2 = time.clock()
        self.lion_time = t2 - t1
        gc.restore_state()
        gc.save_state()

        self.total_paint_time = (self.image_time + self.lion_time + 
                                 self.blit_time + self.clear_time)
        text = "total time: %3.3f" % self.total_paint_time
        gc.set_fill_color((1,1,1))
        gc.show_text(text, (10, sz[1] - 20))
        text = "frames/sec: %3.3f" % (1.0/self.total_paint_time)
        gc.show_text(text, (10, sz[1] - 40))
        gc.restore_state()
        
class BrainCanvasWindow(wx.Frame):
    def __init__(self, id=-1, title='Kiva wxPython Demo',size=(600,800)):
        parent = None
        wx.Frame.__init__(self, parent, id, title, size=size)
        canvas = BrainCanvas(self)
        canvas.SetSize((500,500))
        controls = canvas.controls_ui(parent=self, kind='subpanel')
                
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)
        self.SetAutoLayout(True)
        
        sizer.Add(canvas,1,wx.EXPAND)
        sizer.Add(controls,0,wx.EXPAND)
        
        sizer.Fit(self)
        
        self.Show(1)        

def main():
    
    class MyApp(wx.App):
        def OnInit(self):
            BrainCanvasWindow(size=(700,700))
            return 1
    
    app = MyApp(0)
    app.MainLoop()

if __name__ == "__main__":
    main()
