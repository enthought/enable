#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
# 
# Author: Enthought, Inc.
# Description: <Enthought kiva package component>
#------------------------------------------------------------------------------
# macexample.py

try:
    import W
except ImportError:
    raise Exception('This example works only on Mac')

class TestCanvas(W.Widget):
    def __init__(self, parent=None, id=-1,position = None, size = None):
        #---------------------------------------------------------------------
        # Set up position and size of canvas.  The position_size variable
        # is a 4-tuple, (l,t,r,b).  If l and t are positive, they specify
        # the widget's distance from the left and top edge of the window.  
        # If r and b are *negative*, they specify the widgets distance from 
        # the right and bottom of the window.  If they are positive, they 
        # the width and height of the widget respectively.
        #
        # We'll default to pretty much filling the entire window.
        #---------------------------------------------------------------------
        position_size = [10,10,-10,-30]
        if position:
            position_size[:2] = position
        if size:
            position_size[2:] = size
        W.Widget.__init__(self, position_size)
        self.set_draw_function(default_draw)

        
    def client_gc(self):
        """ Create a GraphicsContext object that is setup with the
            origin and clipping boundaries of this widget.
        """
        if not self._bounds:
            return
        l, t, r, b = self._bounds
        width = r - l
        height = b - t
        x, y = l, b
        raw_gc = CG.CreateCGContextForPort(self._parentwindow.wid)
        gc = GraphicsContext(raw_gc)
        wx, wy, wwidth, wheight = self._parentwindow._bounds
        gc.translate_ctm(x, wheight - y)
        self.border_box = (0, 0, width, height)
        gc.clip_to_rect(self.border_box)

        # initialize to some standard font (necessary?)
        gc.select_font("Times New Roman",12,1)

        return gc
        
    def draw(self, gc, vis_rgn=None):

        if not self._visible:
            return
        
        if not gc:
            print 'oops'
            return 
        gc.save_state()
        #gc.scale_ctm(2,2)
        args = (gc,) + self.draw_args
        self.draw_function(*args,**self.draw_kwargs)
        gc.restore_state()
        self.draw_border(gc)
    
    def draw_border(self, gc):
        gc.stroke_rect_with_width(self.border_box, 0.5)

    def set_draw_function(self,function, *args, **kwargs):
        self.draw_function = function
        self.draw_args = args
        self.draw_kwargs = kwargs
        gc = self.client_gc()
        self.draw(gc)  

class CanvasWindow(W.Window):
    def __init__(self,id=-1,title="Drawing Canvas", size=(300, 300),**kw):
        """ id isn't used on Mac.
        """
        W.Window.__init__(self,size, title,**kw)
        self.canvas = TestCanvas()
        self.open()


                    
if __name__ == "__main__":
    w =test_canvas_window(minsize=(100,100))
    w.canvas.set_draw_function(cap_sampler)
