# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style


include "CoreFoundation.pxi"
include "ATS.pxi"
include "Python.pxi"

cdef class ATSFont:
    cdef ATSFontRef ats_font
    cdef readonly object postscript_name
    def __init__(self, object postscript_name):
        cdef CFStringRef cf_ps_name
        cdef char* c_ps_name

        self.postscript_name = postscript_name
        postscript_name = postscript_name.encode('utf-8')
        c_ps_name = PyString_AsString(postscript_name)
        if c_ps_name == NULL:
            raise ValueError("could not decode %r as a UTF-8 encoded string" % postscript_name)
        cf_ps_name = CFStringCreateWithCString(NULL, c_ps_name,
            kCFStringEncodingUTF8)
        if cf_ps_name == NULL:
            raise ValueError("could not create CFString from %r" % postscript_name)

        self.ats_font = ATSFontFindFromPostScriptName(cf_ps_name,
            kATSOptionFlagsDefault)
        CFRelease(cf_ps_name)
#        if self.ats_font == 0:
#            raise ValueError("could not find font '%s'" % postscript_name)
#        Actually, I have no idea what the error result is
        

    def get_descent_ascent(self, object text):
        """ Get the descent and ascent font metrics for a given string of text.
        """

        cdef ATSFontMetrics metrics
        cdef OSStatus status

        status = ATSFontGetVerticalMetrics(self.ats_font, 0, 
            &metrics)

        return metrics.descent, metrics.ascent


cdef OSStatus families_callback(ATSFontFamilyRef family, void* data):
    cdef CFStringRef cf_fam_name
    cdef OSStatus status
    cdef char* c_fam_name
    cdef char buf[256]
    cdef Boolean success
    
    status = ATSFontFamilyGetName(family, kATSOptionFlagsDefault, &cf_fam_name)

    family_list = <object>data
    c_fam_name = CFStringGetCStringPtr(cf_fam_name, kCFStringEncodingMacRoman)
    if c_fam_name == NULL:
        success = CFStringGetCString(cf_fam_name, buf, 256,
            kCFStringEncodingMacRoman)
        family_list.append(buf)
    else:
        family_list.append(c_fam_name)
    CFRelease(cf_fam_name)
    return status


cdef OSStatus styles_callback(ATSFontRef font, void* data):
    pass


cdef class FontLookup:
    cdef readonly object cache
    cdef public object default_font

    def lookup_ps_name(self, name, style='regular'):
        cdef CFStringRef cf_name
        cdef ATSFontRef ats_font
        cdef OSStatus status

        style = style.title()

        if style == 'Regular':
            style = ''

        mac_name = ' '.join([name, style]).strip().encode('utf-8')
        cf_name = CFStringCreateWithCString(NULL, mac_name,
            kCFStringEncodingUTF8)
        ats_font = ATSFontFindFromName(cf_name, kATSOptionFlagsDefault)
        CFRelease(cf_name)
        if not ats_font:
            raise ValueError("could not find font %r" % mac_name)
        
        status = ATSFontGetPostScriptName(ats_font, kATSOptionFlagsDefault, &cf_name)
        if status != noErr:
            msg = "unknown error getting PostScript name for font %r"%mac_name
            raise RuntimeError(msg)
        
        ps_name = PyString_FromString(CFStringGetCStringPtr(cf_name,
            kCFStringEncodingMacRoman))
        CFRelease(cf_name)
        return ps_name

    def lookup(self, name=None, style='regular'):
        ps_name = self.lookup_ps_name(name or self.default_font, style)
        return ATSFont(ps_name)

    def names(self):
        families = []
        ATSFontFamilyApplyFunction(families_callback, <void*>families)
        families.sort()
        return families

    def styles(self, font_name):
        raise NotImplementedError

    def list_fonts(self):
        for name in self.names():
            print name, self.styles(name)

default_font_info = FontLookup()
default_font_info.default_font = 'Helvetica'
