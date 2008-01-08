import Tkinter
from enthought.kiva import GraphicsContextSystem #, GetDC, ReleaseDC
from enthought import kiva

serengetti = kiva.agg.Image("serengeti.jpg")

root = Tkinter.Tk()

sz = (600,400)

def new_gc(w,h):
    #print w,h    
    gc = GraphicsContextSystem((w,h))
    gc.draw_image(serengetti,(0,0,w,h))
    gc.set_line_width(20)
    gc.set_stroke_color((1,0,0,.3))
    gc.move_to(0,0)
    gc.line_to(w,h)
    gc.stroke_path()
    return gc

def gc_to_window(gc,win):
    hwnd = win.winfo_id()
    hdc = GetDC(hwnd)
    gc.pixel_map.draw(hdc,0,0)
    ReleaseDC(hwnd,hdc)
     
def configure_callback(event):   
    # This leads to flashing, but without it, a paint process
    # is left that results in a blank screen.  Even now, you 
    # occasionally get a blank screen.  Someone more familiar
    # with tkinter will need to help with this.
    event.widget.update()
    w = event.width
    h = event.height
    if (w != event.widget.gc.width() or
        h != event.widget.gc.height()):
        # setup new gc
        gc = new_gc(w,h)
        event.widget.gc = gc
        #gc_to_window(gc,event.widget)
    gc_to_window(event.widget.gc,event.widget)        
        
can = Tkinter.Frame(width=sz[0], height = sz[1])
can.pack(expand=1,fill=Tkinter.BOTH)
can.update()
w = can.winfo_width()
h = can.winfo_height()
can.gc = new_gc(w,h)
gc_to_window(can.gc,can)
can.bind("<Configure>", configure_callback)

root.mainloop()
