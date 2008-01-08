# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style


cdef extern from "Quickdraw.h":
    ctypedef void* CGrafPtr

    ctypedef struct QDRect "Rect":
        short    top
        short    left
        short    bottom
        short    right

    OSStatus CreateCGContextForPort(CGrafPtr inPort, CGContextRef* outContext)
    OSStatus QDBeginCGContext(CGrafPtr inPort, CGContextRef* outContext)
    OSStatus QDEndCGContext(CGrafPtr inPort, CGContextRef* outContext)
    QDRect* GetPortBounds(CGrafPtr port, QDRect* rect)
    OSStatus SyncCGContextOriginWithPort(CGContextRef context, CGrafPtr port)

