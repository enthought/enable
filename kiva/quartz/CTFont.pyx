# (C) Copyright 2004-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from __future__ import print_function

include "CoreFoundation.pxi"
include "CoreGraphics.pxi"
include "CoreText.pxi"


cdef object _cf_string_to_pystring(CFStringRef cf_string):
    cdef char* c_string
    cdef char buf[256]
    c_string = CFStringGetCStringPtr(cf_string, kCFStringEncodingMacRoman)
    if c_string == NULL:
        success = CFStringGetCString(cf_string, buf, 256,
                    kCFStringEncodingMacRoman)
        retval = str(buf)
    else:
        retval = str(c_string)
    return retval

cdef CFArrayRef _get_system_fonts():
        cdef CFIndex value = 1
        cdef CFNumberRef cf_number
        cdef CFMutableDictionaryRef cf_options_dict
        cdef CTFontCollectionRef cf_font_collection
        cdef CFArrayRef cf_font_descriptors

        cf_options_dict = CFDictionaryCreateMutable(NULL, 0,
                            &kCFTypeDictionaryKeyCallBacks,
                            &kCFTypeDictionaryValueCallBacks)

        if cf_options_dict != NULL:
            cf_number = CFNumberCreate(NULL, kCFNumberCFIndexType, &value)
            CFDictionaryAddValue(cf_options_dict, <void*>kCTFontCollectionRemoveDuplicatesOption,
                <void*>cf_number)
            CFRelease(cf_number)
        else:
            msg = "unknown error building options dictionary for font list"
            raise RuntimeError(msg)

        cf_font_collection = CTFontCollectionCreateFromAvailableFonts(cf_options_dict)
        CFRelease(cf_options_dict)

        cf_font_descriptors = CTFontCollectionCreateMatchingFontDescriptors(cf_font_collection)
        CFRelease(cf_font_collection)

        return cf_font_descriptors

cdef class CTFont:
    cdef CTFontRef ct_font
    def __cinit__(self, *args, **kwargs):
        self.ct_font = NULL

    def __dealloc__(self):
        if self.ct_font != NULL:
            CFRelease(self.ct_font)

    cpdef size_t get_pointer(self):
        return <size_t>self.ct_font

    cdef set_pointer(self, CTFontRef pointer):
        self.ct_font = pointer


cdef class CTFontStyle:
    cdef CTFontDescriptorRef ct_font_descriptor
    cdef CFDictionaryRef attribute_dictionary
    cdef readonly object family_name
    cdef readonly object style
    
    def __cinit__(self, *args, **kwargs):
        self.ct_font_descriptor = NULL
        self.attribute_dictionary = NULL

    def __dealloc__(self):
        if self.attribute_dictionary != NULL:
            CFRelease(self.attribute_dictionary)
        if self.ct_font_descriptor != NULL:
            CFRelease(self.ct_font_descriptor)

    def __init__(self, name, style='regular'):
        self.family_name = name
        self.style = style
        self.attribute_dictionary = self._build_attribute_dictionary(name, style)
        self.ct_font_descriptor = CTFontDescriptorCreateWithAttributes(self.attribute_dictionary)

    def get_descent_ascent(self, object text):
        """ Get the descent and ascent font metrics for a given string of text.
        """

        cdef CTFontRef ct_font
        cdef float ascent, descent

        ct_font = CTFontCreateWithFontDescriptor(self.ct_font_descriptor, 0.0, NULL)
        ascent = CTFontGetAscent(ct_font)
        descent = CTFontGetDescent(ct_font)
        CFRelease(ct_font)

        return descent, ascent

    def get_font(self, float font_size):
        """ Get a CTFont matching the descriptor at the given size.
        """
        cdef CTFontRef ct_font
        
        ct_font = CTFontCreateWithFontDescriptor(self.ct_font_descriptor,
                    font_size, NULL)
        font = CTFont()
        font.set_pointer(ct_font)
        return font

    property postcript_name:
        def __get__(self):
            cdef CFStringRef cf_ps_name
            cf_ps_name = <CFStringRef>CTFontDescriptorCopyAttribute(self.ct_font_descriptor,
                                        kCTFontNameAttribute)
            retval = _cf_string_to_pystring(cf_ps_name)
            CFRelease(cf_ps_name)

            return retval

    cdef CFDictionaryRef _build_attribute_dictionary(self, name, style):
        cdef CFStringRef cf_name, cf_style
        cdef CFMutableDictionaryRef cf_dict

        mac_name = name.strip().encode('utf-8')
        cf_name = CFStringCreateWithCString(NULL, mac_name,
            kCFStringEncodingUTF8)

        style = style.title()
        mac_style = style.strip().encode('utf-8')
        cf_style = CFStringCreateWithCString(NULL, mac_style,
            kCFStringEncodingUTF8)

        cf_dict = CFDictionaryCreateMutable(NULL, 0,
                    &kCFTypeDictionaryKeyCallBacks,
                    &kCFTypeDictionaryValueCallBacks)

        if cf_dict != NULL:
            CFDictionaryAddValue(cf_dict, <void*>kCTFontFamilyNameAttribute, cf_name)
            CFDictionaryAddValue(cf_dict, <void*>kCTFontStyleNameAttribute, cf_style)
        else:
            msg = "unknown error building descriptor dictionary for font %r"%mac_name
            raise RuntimeError(msg)

        CFRelease(cf_name)
        CFRelease(cf_style)

        return cf_dict


cdef class FontLookup:
    cdef readonly object cache
    cdef public object default_font

    def lookup(self, name=None, style='regular'):
        return CTFontStyle(name or self.default_font, style)

    def names(self):
        cdef CFIndex idx, count
        cdef CFStringRef cf_fam_name
        cdef CFArrayRef cf_font_descriptors
        cdef CTFontDescriptorRef cf_font_descriptor

        cf_font_descriptors = _get_system_fonts()
        families = []
        count = CFArrayGetCount(cf_font_descriptors)
        for idx from 0 <= idx < count:
            cf_font_descriptor = <CTFontDescriptorRef>CFArrayGetValueAtIndex(cf_font_descriptors, idx)
            cf_fam_name = <CFStringRef>CTFontDescriptorCopyAttribute(cf_font_descriptor,
                                        kCTFontFamilyNameAttribute)
            families.append(_cf_string_to_pystring(cf_fam_name))
            CFRelease(cf_fam_name)
        CFRelease(cf_font_descriptors)

        families.sort()
        return list(set(families))

    def styles(self, font_name):
        raise NotImplementedError

    def list_fonts(self):
        for name in self.names():
            print(name, self.styles(name))

default_font_info = FontLookup()
default_font_info.default_font = 'Helvetica'
