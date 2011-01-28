# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style


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

    ctypedef CFTypeRef CFStringRef

    ctypedef unsigned int CFIndex
    ctypedef struct CFRange:
        CFIndex location
        CFIndex length

    CFRange CFRangeMake(CFIndex location, CFIndex length)

    CFStringRef CFStringCreateWithCString(void* alloc, char* cStr,
        CFStringEncoding encoding)
    char* CFStringGetCStringPtr(CFStringRef string, CFStringEncoding encoding)
    Boolean CFStringGetCString(CFStringRef theString, char* buffer,
        CFIndex bufferSize, CFStringEncoding encoding)
    void CFRelease(CFTypeRef cf)
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

    ctypedef CFTypeRef CFDictionaryRef


