import os, wx
from enthought.kiva import *
from enthought.kiva.backend_image import Image
from enthought.traits.api import Trait, HasTraits, TraitRange, Bool
from numpy import array, arange, sin, pi

import time
import lion_data


class LionModel(HasTraits):
    # fix me: Split this out since the LionCanvas cannot derive from 
    # HasTraits and a wxPython 2.4 window (new style class issue).  
    x = Trait(0.36,TraitRange(-0.5,0.5))
    y = Trait(0.5,TraitRange(-0.5,0.5))
    rotation = Trait(0.0,TraitRange(0.0,360.0))
    scale = Trait(0.59,TraitRange(0.1,3.0))
    opacity = Trait(1.0,TraitRange(0.0,1.0))
    use_image = Bool(True)
    antialias = Bool(True)
    
    def __init__(self, *args, **kw):
        HasTraits.__init__(self, *args, **kw)
        self.x_changed_time = 0.0
    
    def _x_changed(self,value):
        self.x_changed_time = time.clock()
        print 'x_change x:', value
        
class LionCanvas(Canvas):

    def __init__(self, parent, id = -1, size = wx.DefaultSize):
        Canvas.__init__(self, parent,id,size=size)
        path_and_color, size, center = lion_data.get_lion()
        self.path_and_color = path_and_color

        dir_name = os.path.dirname(os.path.abspath(__file__))
        self.raw_serengeti = Image(os.path.join(dir_name, "serengeti.jpg"),
                                   interpolation="nearest")
        self._create_image(size)
        
        self.model = LionModel()
        self.model.on_trait_change(self.update_window)
        
        self.font = Font(face_name="Arial")
        
        self.last_time = time.clock()
        
        curves = 10
        N=10000
        self.lines = []
        for i in range(curves):
            # vertical rendering
            x = arange(N) * 0.1 + 100
            y = sin(x*pi/N*1000.0)*50 #+ random(size=x.shape)*40.0
            self.lines.append(array(zip(y,x)))
    
    def _create_image(self, size):
        size = tuple(size)
        self.serengeti = self._create_kiva_gc(size)
        self.serengeti.draw_image(self.raw_serengeti, (0, 0, size[0], size[1]))
        #self.serengeti = self.raw_serengeti
                        
        return
                        
    def controls_ui(self, parent=None, kind='modal'):
        return self.model.edit_traits(parent=parent, kind=kind).control
    
    def OnSize(self,event):
        Canvas.OnSize(self, event)
        size = self.GetClientSizeTuple()
        self._create_image(size)

        return
        
    def update_window(self, obj, name, old, new):
        self.dirty=1
        self.Refresh()
        #self.Update()

    def do_draw(self,gc):
        
        print 'do draw x:', self.model.x
        now = time.clock()
        call_time = now - self.last_time
        self.last_time = now
        print 'time difference between x set and draw:', now - self.model.x_changed_time, 1/(now - self.model.x_changed_time)
        
        sz = self.GetClientSizeTuple()
        gc.save_state()
        gc.set_font(self.font)
        #gc.rotate_ctm(3.1416/4)
        t1 = time.clock()
        if self.model.use_image:
            gc.save_state()
            # fix me: need to blend modes set up as kiva constants
            from enthought.kiva.agg import blend_copy
            gc.set_blend_mode(blend_copy)
            gc.draw_image(self.serengeti,(0,0,sz[0],sz[1]))
            gc.restore_state()
            
        #if self.model.use_image:
        #    gc.copy_image(self.serengeti,0,0)
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
        
        gc.set_alpha(self.model.opacity)
        gc.set_antialias(self.model.antialias)
        
        for line in self.lines:
            
            gc.begin_path()
            gc.lines(line)
        
            gc.stroke_path()
            gc.translate_ctm(100,0)
            
        #for path,color in self.path_and_color:            
        #    gc.begin_path()
        #    gc.add_path(path)
        #    gc.set_fill_color(color)
        #    gc.set_alpha(self.model.opacity)
        #    gc.fill_path()       
        t2 = time.clock()
        self.lion_time = t2 - t1
        gc.restore_state()
        gc.save_state()
        self.total_paint_time = (self.image_time + self.lion_time + 
                                 self.blit_time + self.clear_time)
                                 
        if not gc.is_font_initialized():
            f = Font(face_name="Arial")
            gc.set_font(f)
            
        text = "total time: %3.3f" % self.total_paint_time
        gc.show_text(text,(10,sz[1] - 20))
        text = "drawing frames/sec: %3.3f" % (1.0/self.total_paint_time)
        gc.show_text(text,(10,sz[1] - 40))
        text = "actual frames/sec: %3.3f" % (1.0/call_time)
        gc.show_text(text,(10,sz[1] - 60))
        now = time.clock()
        text = "x change frames/sec: %3.3f" % (1/(now - self.model.x_changed_time))
        gc.show_text(text,(10,sz[1] - 80))
        text = "clear time: %3.3f" % (self.clear_time)
        gc.show_text(text,(10,sz[1] - 100))
        text = "blit time: %3.3f, %3.3f" % (self.blit_time, 1/(self.blit_time+.00001))
        gc.show_text(text,(10,sz[1] - 120))
        text = "image time: %3.3f, %3.3f" % (self.image_time, 1/(self.image_time+.00001))
        gc.show_text(text,(10,sz[1] - 140))
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
