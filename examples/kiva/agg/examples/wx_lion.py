import wx
from enthought.kiva import *
from enthought.kiva.backend_image import Image#, FontType
from enthought.traits.api import Bool, HasTraits, TraitRange, Trait
from numpy  import array

import time
import lion_data

def fixpath(path):
    import os
    return os.path.join(os.path.dirname(__file__),path)

class LionModel(HasTraits):
    # fix me: Split this out since the LionCanvas cannot derive from 
    # HasTraits and a wxPython 2.4 window (new style class issue).  
    x = Trait(0.15,TraitRange(-0.5,0.5))
    y = Trait(-0.19,TraitRange(-0.5,0.5))
    rotation = Trait(0.0,TraitRange(0.0,360.0))
    scale = Trait(0.35,TraitRange(0.1,3.0))
    opacity = Trait(1.0,TraitRange(0.0,1.0))
    use_image = Bool(True)
    
class LionCanvas(Canvas):

    def __init__(self, parent, id = -1, size = wx.DefaultSize):
        Canvas.__init__(self, parent,id,size=size)
        path_and_color, size, center = lion_data.get_lion()
        self.path_and_color = path_and_color
        self.serengeti = Image(fixpath("serengeti.jpg"), interpolation="nearest")

        self.model = LionModel()
        self.model.on_trait_change(self.update_window)
        
        self.font = Font(face_name="Arial")
        
    def controls_ui(self, parent=None, kind='modal'):
        return self.model.edit_traits(parent=parent, kind=kind).control
            
    def update_window(self):
        self.dirty=1
        self.Refresh()

    def do_draw(self,gc):
        sz = self.GetClientSizeTuple()
        gc.save_state()
        #gc.rotate_ctm(3.1416/4)
        t1 = time.clock()
        if self.model.use_image:
            gc.draw_image(self.serengeti,(0,0,sz[0],sz[1]))
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
        gc.scale_ctm(self.model.scale,self.model.scale)
        for path,color in self.path_and_color:            
            gc.begin_path()
            gc.add_path(path)
            gc.set_fill_color(color)
            gc.set_alpha(self.model.opacity)
            gc.fill_path()       
        t2 = time.clock()
        self.lion_time = t2 - t1
        gc.restore_state()
        gc.save_state()
        gc.set_font(self.font)
        self.total_paint_time = (self.image_time + self.lion_time + 
                                 self.blit_time + self.clear_time)
        text = "total time: %3.3f" % self.total_paint_time
        gc.show_text(text,(10,sz[1] - 20))
        if self.total_paint_time > 1E-8:
            text = "frames/sec: %3.3f" % (1.0/self.total_paint_time)
        else:
            text = "frames/sec: -----"
        gc.show_text(text,(10,sz[1] - 40))
        gc.restore_state()
        
class LionCanvasWindow(wx.Frame):
    def __init__(self, id=-1, title='Lion Example',size=(600,800)):
        parent = None
        wx.Frame.__init__(self, parent, id, title, size=size)
        
        canvas = LionCanvas(self)
        canvas.SetSize((500,500))
        controls = canvas.controls_ui(parent=self, kind='subpanel')
                
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)
        self.SetAutoLayout(True)
        
        # fix me: canvas is not accurately reporting its size...
        sizer.Add(canvas,1,wx.EXPAND)
        sizer.Add(controls,0,wx.EXPAND)
        
        sizer.Fit(self)
        
        self.Show(1)        

def main():
    
    class MyApp(wx.App):
        def OnInit(self):
            LionCanvasWindow(size=(700,700))
            return 1
    
    app = MyApp(0)
    app.MainLoop()

if __name__ == "__main__":
    main()
