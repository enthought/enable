# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style



cdef class ATSFont:
    cdef ATSFontRef ats_font
    cdef readonly object postscript_name

cdef class FontLookup:
    cdef readonly object cache
    cdef public object default_font

