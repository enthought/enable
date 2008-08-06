"""

"""
import math
import wx

import enthought.savage.svg.svg_extras

def AddEllipticalArc(self, x, y, w, h, start_angle, extent, clockwise=False):
    """ Draws an arc of an ellipse within bounding rect (x,y,w,h) 
    from startArc to endArc (in degrees, relative to the horizontal line of the eclipse)"""

    # compute the cubic bezier and add that to the path by calling AddCurveToPoint
    sub_paths = svg_extras.bezier_arc(x, y, x+w, y+h, start_angle, extent)
    for sub_path in sub_paths:
	    x1,y1, cx1, cy1, cx2, cy2, x2,y2 = sub_path
            
            path = wx.GraphicsRenderer_GetDefaultRenderer().CreatePath()
	    path.MoveToPoint(x1, y1)
	    path.AddCurveToPoint(cx1, cy1, cx2, cy2, x2, y2)
            
            self.AddPath(path)
            self.MoveToPoint(path.GetCurrentPoint())
            self.CloseSubpath()
    
if not hasattr(wx.GraphicsPath, "AddEllipticalArcTo"):
    wx.GraphicsPath.AddEllipticalArcTo = AddEllipticalArc

del AddEllipticalArc
    
