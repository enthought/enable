# (C) Copyright 2004-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

cdef extern from "CoreFoundation/CoreFoundation.h":
    ctypedef int OSStatus

    cdef enum:
        noErr

    ctypedef enum CFStringEncoding:
        kCFStringEncodingMacRoman = 0
        kCFStringEncodingWindowsLatin1 = 0x0500
        kCFStringEncodingISOLatin1 = 0x0201
        kCFStringEncodingNextStepLatin = 0x0B01
        kCFStringEncodingASCII = 0x0600
        kCFStringEncodingUnicode = 0x0100
        kCFStringEncodingUTF8 = 0x08000100
        kCFStringEncodingNonLossyASCII = 0x0BFF

    ctypedef unsigned char UInt8
    ctypedef unsigned short UniChar
    ctypedef int bool
    ctypedef bool Boolean

    ctypedef void* CFTypeRef
    ctypedef unsigned int CFTypeID

    void CFRelease(CFTypeRef cf)

    ctypedef CFTypeRef CFStringRef

    ctypedef unsigned int CFIndex
    ctypedef CFIndex CFNumberType

    ctypedef unsigned long CFHashCode
    ctypedef struct CFRange:
        CFIndex location
        CFIndex length

    CFRange CFRangeMake(CFIndex location, CFIndex length)

    CFStringRef CFStringCreateWithCString(void* alloc, char* cStr,
        CFStringEncoding encoding)
    char* CFStringGetCStringPtr(CFStringRef string, CFStringEncoding encoding)
    Boolean CFStringGetCString(CFStringRef theString, char* buffer,
        CFIndex bufferSize, CFStringEncoding encoding)
    CFIndex CFStringGetLength(CFStringRef theString)
    void CFStringGetCharacters(CFStringRef theString, CFRange range, UniChar *buffer)

    ctypedef enum CFURLPathStyle:
        kCFURLPOSIXPathStyle = 0
        kCFURLHFSPathStyle = 1
        kCFURLWindowsPathStyle = 2

    ctypedef CFTypeRef CFURLRef

    CFURLRef CFURLCreateWithFileSystemPath(void* allocator,
        CFStringRef filePath, CFURLPathStyle pathStyle, bool isDirectory)
    void CFShow(CFTypeRef cf)
    CFTypeID CFGetTypeID(CFTypeRef cf)
    
    ctypedef struct CFArrayCallBacks:
        CFIndex version
        #CFArrayRetainCallBack retain
        #CFArrayReleaseCallBack release
        #CFArrayCopyDescriptionCallBack copyDescription
        #CFArrayEqualCallBack equal
    
    cdef CFArrayCallBacks kCFTypeArrayCallBacks
    #ctypedef void (*CFArrayApplierFunction)(void *value, void *context)
    ctypedef CFTypeRef CFArrayRef
    ctypedef CFTypeRef CFMutableArrayRef
    
    CFArrayRef CFArrayCreate(void* allocator, void **values,
        CFIndex numValues, CFArrayCallBacks *callBacks)
    CFArrayRef CFArrayCreateCopy(void* allocator, CFArrayRef theArray)
    CFMutableArrayRef CFArrayCreateMutable(void* allocator, CFIndex capacity,
        CFArrayCallBacks *callBacks)
    CFMutableArrayRef CFArrayCreateMutableCopy(void* allocator, CFIndex capacity,
        CFArrayRef theArray)
    CFIndex CFArrayGetCount(CFArrayRef theArray)
    CFIndex CFArrayGetCountOfValue(CFArrayRef theArray, CFRange range,
        void *value)
    Boolean CFArrayContainsValue(CFArrayRef theArray, CFRange range,
        void *value)
    void *CFArrayGetValueAtIndex(CFArrayRef theArray, CFIndex idx)
    void CFArrayGetValues(CFArrayRef theArray, CFRange range, void **values)
    #void CFArrayApplyFunction(CFArrayRef theArray, CFRange range,
    #    CFArrayApplierFunction applier, void *context)
    CFIndex CFArrayGetFirstIndexOfValue(CFArrayRef theArray, CFRange range,
        void *value)
    CFIndex CFArrayGetLastIndexOfValue(CFArrayRef theArray, CFRange range,
        void *value)
    #CFIndex CFArrayBSearchValues(CFArrayRef theArray, CFRange range,
    #    void *value, CFComparatorFunction comparator, void *context)
    void CFArrayAppendValue(CFMutableArrayRef theArray, void *value)
    void CFArrayInsertValueAtIndex(CFMutableArrayRef theArray, CFIndex idx,
        void *value)
    void CFArraySetValueAtIndex(CFMutableArrayRef theArray, CFIndex idx,
        void *value)
    void CFArrayRemoveValueAtIndex(CFMutableArrayRef theArray, CFIndex idx)
    void CFArrayRemoveAllValues(CFMutableArrayRef theArray)
    void CFArrayReplaceValues(CFMutableArrayRef theArray, CFRange range,
        void **newValues, CFIndex newCount)
    void CFArrayExchangeValuesAtIndices(CFMutableArrayRef theArray,
        CFIndex idx1, CFIndex idx2)
    #void CFArraySortValues(CFMutableArrayRef theArray, CFRange range,
    #    CFComparatorFunction comparator, void *context)
    void CFArrayAppendArray(CFMutableArrayRef theArray, CFArrayRef otherArray,
        CFRange otherRange)


    ctypedef CFTypeRef CFDictionaryRef
    ctypedef CFTypeRef CFMutableDictionaryRef

    ctypedef struct CFDictionaryKeyCallBacks:
        CFIndex version
        #CFDictionaryRetainCallBack retain
        #CFDictionaryReleaseCallBack release
        #CFDictionaryCopyDescriptionCallBack copyDescription
        #CFDictionaryEqualCallBack equal
        #CFDictionaryHashCallBack hash
    
    ctypedef struct CFDictionaryValueCallBacks:
        CFIndex version
        #CFDictionaryRetainCallBack retain
        #CFDictionaryReleaseCallBack release
        #CFDictionaryCopyDescriptionCallBack copyDescription
        #CFDictionaryEqualCallBack equal

    cdef CFDictionaryKeyCallBacks kCFTypeDictionaryKeyCallBacks
    cdef CFDictionaryValueCallBacks kCFTypeDictionaryValueCallBacks

    CFDictionaryRef CFDictionaryCreate(void* allocator,
        void** keys, void** values, CFIndex numValues,
        CFDictionaryKeyCallBacks* keyCallBacks,
        CFDictionaryValueCallBacks* valueCallBacks)    
    CFMutableDictionaryRef CFDictionaryCreateMutable(void* allocator,
        CFIndex capacity, CFDictionaryKeyCallBacks *keyCallBacks,
        CFDictionaryValueCallBacks *valueCallBacks)
    void CFDictionaryAddValue(CFMutableDictionaryRef theDict, void *key, void *value)
    void CFDictionarySetValue(CFMutableDictionaryRef theDict, void *key, void *value)

    ctypedef CFTypeRef CFAttributedStringRef
    ctypedef CFTypeRef CFMutableAttributedStringRef

    CFAttributedStringRef CFAttributedStringCreate(void* alloc,
        CFStringRef str, CFDictionaryRef attributes)
    CFMutableAttributedStringRef CFAttributedStringCreateMutable(void* alloc,
        CFIndex maxLength)
    void CFAttributedStringReplaceString(CFMutableAttributedStringRef aStr,
        CFRange range, CFStringRef replacement)
    void CFAttributedStringSetAttribute(CFMutableAttributedStringRef aStr,
        CFRange range, CFStringRef attrName, CFTypeRef value)

    ctypedef CFTypeRef CFNumberRef
    
    ctypedef enum CFNumberType_:
        kCFNumberSInt8Type = 1
        kCFNumberSInt16Type = 2
        kCFNumberSInt32Type = 3
        kCFNumberSInt64Type = 4
        kCFNumberFloat32Type = 5
        kCFNumberFloat64Type = 6
        kCFNumberCharType = 7
        kCFNumberShortType = 8
        kCFNumberIntType = 9
        kCFNumberLongType = 10
        kCFNumberLongLongType = 11
        kCFNumberFloatType = 12
        kCFNumberDoubleType = 13
        kCFNumberCFIndexType = 14

    CFNumberRef CFNumberCreate(void* allocator, CFNumberType theType, void *valuePtr)





