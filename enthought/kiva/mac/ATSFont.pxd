# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style


include "CoreFoundation.pxi"
include "ATS.pxi"

cdef class ATSFont:
    cdef ATSFontRef ats_font
    cdef readonly object postscript_name

cdef class FontLookup:
    cdef readonly object cache
    cdef public object default_font

