# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style


cdef CFURLRef url_from_filename(char* filename) except NULL:
    cdef CFStringRef filePath
    filePath = CFStringCreateWithCString(NULL, filename,
        kCFStringEncodingUTF8)
    if filePath == NULL:
        raise RuntimeError("could not create CFStringRef")

    cdef CFURLRef cfurl
    cfurl = CFURLCreateWithFileSystemPath(NULL, filePath,
        kCFURLPOSIXPathStyle, 0)
    CFRelease(filePath)
    if cfurl == NULL:
        raise RuntimeError("could not create a CFURLRef")
    return cfurl

