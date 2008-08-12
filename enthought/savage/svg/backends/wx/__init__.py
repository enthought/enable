"""

"""
import math
import wx

from enthought.savage.svg import svg_extras

def elliptical_arc_to(self, rx, ry, phi, large_arc_flag, sweep_flag, x2, y2):
    pass
    

def AddEllipticalArc(self, x, y, width, height, theta, dtheta, clockwise=False):
    """ Draws an arc of an ellipse within bounding rect (x,y,w,h) 
    from startArc to endArc (in degrees, relative to the horizontal line of the eclipse)"""

    # compute the cubic bezier and add that to the path by calling AddCurveToPoint
    sub_paths = svg_extras.bezier_arc(x, y, x+width, y+height, theta, dtheta)
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
    wx.GraphicsPath.elliptical_arc_to = elliptical_arc_to

del AddEllipticalArc
    
