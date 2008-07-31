# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style


cdef extern from "ApplicationServices/ApplicationServices.h":
    ctypedef void*  CGContextRef

    ctypedef struct CGPoint:
        float x
        float y

    ctypedef struct CGSize:
        float width
        float height

    ctypedef struct CGRect:
        CGPoint origin
        CGSize size

    ctypedef struct CGAffineTransform:
        float a
        float b
        float c
        float d
        float tx
        float ty

    ctypedef enum CGBlendMode:
        kCGBlendModeNormal
        kCGBlendModeMultiply
        kCGBlendModeScreen
        kCGBlendModeOverlay
        kCGBlendModeDarken
        kCGBlendModeLighten
        kCGBlendModeColorDodge
        kCGBlendModeColorBurn
        kCGBlendModeSoftLight
        kCGBlendModeHardLight
        kCGBlendModeDifference
        kCGBlendModeExclusion
        kCGBlendModeHue
        kCGBlendModeSaturation
        kCGBlendModeColor
        kCGBlendModeLuminosity 

    ctypedef enum CGLineCap:
        kCGLineCapButt
        kCGLineCapRound
        kCGLineCapSquare

    ctypedef enum CGLineJoin:
        kCGLineJoinMiter
        kCGLineJoinRound
        kCGLineJoinBevel

    ctypedef enum CGPathDrawingMode:
        kCGPathFill
        kCGPathEOFill
        kCGPathStroke
        kCGPathFillStroke
        kCGPathEOFillStroke

    ctypedef enum CGRectEdge:
        CGRectMinXEdge
        CGRectMinYEdge
        CGRectMaxXEdge
        CGRectMaxYEdge

    # CGContext management
    CGContextRef CGContextRetain(CGContextRef context)
    CGContextRef CGPDFContextCreateWithURL(CFURLRef url, CGRect * mediaBox,
        CFDictionaryRef auxiliaryInfo)
    void CGContextRelease(CGContextRef context)
    void CGContextFlush(CGContextRef context)
    void CGContextSynchronize(CGContextRef context)
    void CGContextBeginPage(CGContextRef context, CGRect* mediaBox)
    void CGContextEndPage(CGContextRef context)

    # User-space transformations
    void CGContextScaleCTM(CGContextRef context, float sx, float sy)
    void CGContextTranslateCTM(CGContextRef context, float tx, float ty)
    void CGContextRotateCTM(CGContextRef context, float angle)
    void CGContextConcatCTM(CGContextRef context, CGAffineTransform transform)
    CGAffineTransform CGContextGetCTM(CGContextRef context)

    # Saving and Restoring the Current Graphics State
    void CGContextSaveGState(CGContextRef context)
    void CGContextRestoreGState(CGContextRef context)

    # Changing the Current Graphics State
    void CGContextSetFlatness(CGContextRef context, float flatness)
    void CGContextSetLineCap(CGContextRef context, CGLineCap cap)
    void CGContextSetLineDash(CGContextRef context, float phase,
        float lengths[], size_t count)
    void CGContextSetLineJoin(CGContextRef context, CGLineJoin join)
    void CGContextSetLineWidth(CGContextRef context, float width)
    void CGContextSetMiterLimit(CGContextRef context, float limit)
    void CGContextSetPatternPhase(CGContextRef context, CGSize phase)
    void CGContextSetShouldAntialias(CGContextRef context, bool shouldAntialias)
    void CGContextSetShouldSmoothFonts(CGContextRef context, bool shouldSmoothFonts)
    void CGContextSetAllowsAntialiasing(CGContextRef context, bool allowsAntialiasing)

    # Constructing Paths
    void CGContextBeginPath(CGContextRef context)
    void CGContextMoveToPoint(CGContextRef context, float x, float y)
    void CGContextAddLineToPoint(CGContextRef context, float x, float y)
    void CGContextAddLines(CGContextRef context, CGPoint points[], size_t count)
    void CGContextAddCurveToPoint(CGContextRef context, float cp1x, float cp1y,
        float cp2x, float cp2y, float x, float y)
    void CGContextAddQuadCurveToPoint(CGContextRef context, float cpx, 
        float cpy, float x, float y)
    void CGContextAddRect(CGContextRef context, CGRect rect)
    void CGContextAddRects(CGContextRef context, CGRect rects[], size_t count)
    void CGContextAddArc(CGContextRef context, float x, float y, float radius,
        float startAngle, float endAngle, int clockwise)
    void CGContextAddArcToPoint(CGContextRef context, float x1, float y1, 
        float x2, float y2, float radius)
    void CGContextClosePath(CGContextRef context)
    void CGContextAddEllipseInRect(CGContextRef context, CGRect rect)

    # Painting Paths
    void CGContextDrawPath(CGContextRef context, CGPathDrawingMode mode)
    void CGContextStrokePath(CGContextRef context)
    void CGContextFillPath(CGContextRef context)
    void CGContextEOFillPath(CGContextRef context)
    void CGContextStrokeRect(CGContextRef context, CGRect rect)
    void CGContextStrokeRectWithWidth(CGContextRef context, CGRect rect, float width)
    void CGContextFillRect(CGContextRef context, CGRect rect)
    void CGContextFillRects(CGContextRef context, CGRect rects[], size_t count)
    void CGContextClearRect(CGContextRef context, CGRect rect)
    void CGContextReplacePathWithStrokedPath(CGContextRef c)
    void CGContextFillEllipseInRect(CGContextRef context, CGRect rect)
    void CGContextStrokeEllipseInRect(CGContextRef context, CGRect rect)
    void CGContextStrokeLineSegments(CGContextRef c, CGPoint points[], size_t count)
 
    # Querying Paths
    int CGContextIsPathEmpty(CGContextRef context)
    CGPoint CGContextGetPathCurrentPoint(CGContextRef context)
    CGRect CGContextGetPathBoundingBox(CGContextRef context)
    bool CGContextPathContainsPoint(CGContextRef context, CGPoint point, CGPathDrawingMode mode)
 
    # Modifying Clipping Paths
    void CGContextClip(CGContextRef context)
    void CGContextEOClip(CGContextRef context)
    void CGContextClipToRect(CGContextRef context, CGRect rect)
    void CGContextClipToRects(CGContextRef context, CGRect rects[], size_t count)
    #void CGContextClipToMask(CGContextRef c, CGRect rect, CGImageRef mask)

    # Color Spaces
    ctypedef void* CGColorSpaceRef

    ctypedef enum CGColorRenderingIntent:
        kCGRenderingIntentDefault
        kCGRenderingIntentAbsoluteColorimetric
        kCGRenderingIntentRelativeColorimetric
        kCGRenderingIntentPerceptual
        kCGRenderingIntentSaturation

    ctypedef enum:
        kCGColorSpaceUserGray
        kCGColorSpaceUserRGB
        kCGColorSpaceUserCMYK

    ctypedef enum FakedEnums:
        kCGColorSpaceGenericGray
        kCGColorSpaceGenericRGB
        kCGColorSpaceGenericCMYK

    CGColorSpaceRef CGColorSpaceCreateDeviceGray()
    CGColorSpaceRef CGColorSpaceCreateDeviceRGB()
    CGColorSpaceRef CGColorSpaceCreateDeviceCMYK()
    CGColorSpaceRef CGColorSpaceCreateWithName(FakedEnums name)
    CGColorSpaceRef CGColorSpaceRetain(CGColorSpaceRef cs)
    int CGColorSpaceGetNumberOfComponents(CGColorSpaceRef cs)
    void CGColorSpaceRelease(CGColorSpaceRef cs) 

    ctypedef void* CGColorRef

    # Color Settings - Partial
    CGColorRef CGColorCreate(CGColorSpaceRef colorspace, float components[])
#    CGColorRef CGColorCreateWithPattern(CGColorSpaceRef colorspace, CGPatternRef pattern, float components[])
    CGColorRef CGColorCreateCopy(CGColorRef color) 
    CGColorRef CGColorCreateCopyWithAlpha(CGColorRef color, float alpha)
    CGColorRef CGColorRetain(CGColorRef color)
    void CGColorRelease(CGColorRef color)
    bool CGColorEqualToColor(CGColorRef color1, CGColorRef color2)
    size_t CGColorGetNumberOfComponents(CGColorRef color)
    float *CGColorGetComponents(CGColorRef color)
    float CGColorGetAlpha(CGColorRef color)
    CGColorSpaceRef CGColorGetColorSpace(CGColorRef color)
#    CGPatternRef CGColorGetPattern(CGColorRef color)

    void CGContextSetFillColorSpace(CGContextRef context, CGColorSpaceRef colorspace)
    void CGContextSetFillColor(CGContextRef context, float components[])
    void CGContextSetStrokeColorSpace(CGContextRef context, 
        CGColorSpaceRef colorspace)
    void CGContextSetStrokeColor(CGContextRef context, float components[])
    void CGContextSetGrayFillColor(CGContextRef context, float gray, float alpha)
    void CGContextSetGrayStrokeColor(CGContextRef context, float gray, float alpha)
    void CGContextSetRGBFillColor(CGContextRef context, float red, float green,
        float blue, float alpha)
    void CGContextSetRGBStrokeColor(CGContextRef context, float red, float green, 
        float blue, float alpha)
    void CGContextSetCMYKFillColor(CGContextRef context, float cyan, float magenta,
        float yellow, float black, float alpha)
    void CGContextSetCMYKStrokeColor(CGContextRef context, float cyan, float magenta,
        float yellow, float black, float alpha)
    void CGContextSetAlpha(CGContextRef context, float alpha)
    void CGContextSetRenderingIntent(CGContextRef context, 
        CGColorRenderingIntent intent)
    void CGContextSetBlendMode(CGContextRef context, CGBlendMode mode)

    # Using ATS Fonts

    ctypedef void* CGFontRef
    ctypedef unsigned short CGFontIndex
    ctypedef CGFontIndex CGGlyph

    cdef enum:
        kCGFontIndexMax = ((1 << 16) - 2)
        kCGFontIndexInvalid = ((1 << 16) - 1)
        kCGGlyphMax = kCGFontIndexMax

    CGFontRef CGFontCreateWithPlatformFont(void *platformFontReference)
    CGFontRef CGFontRetain(CGFontRef font)
    void CGFontRelease(CGFontRef font)

    # Drawing Text
    
    ctypedef enum CGTextDrawingMode:
        kCGTextFill
        kCGTextStroke
        kCGTextFillStroke
        kCGTextInvisible
        kCGTextFillClip
        kCGTextStrokeClip
        kCGTextFillStrokeClip
        kCGTextClip

    ctypedef enum CGTextEncoding:
        kCGEncodingFontSpecific
        kCGEncodingMacRoman

    void CGContextSelectFont(CGContextRef context, char* name, float size, 
        CGTextEncoding textEncoding)
    void CGContextSetFontSize(CGContextRef context, float size)
    void CGContextSetCharacterSpacing(CGContextRef context, float spacing)
    void CGContextSetTextDrawingMode(CGContextRef context, CGTextDrawingMode mode)
    void CGContextSetTextPosition(CGContextRef context, float x, float y)
    CGPoint CGContextGetTextPosition(CGContextRef context)
    void CGContextSetTextMatrix(CGContextRef context, CGAffineTransform transform)
    CGAffineTransform CGContextGetTextMatrix(CGContextRef context)
    void CGContextShowText(CGContextRef context, char* bytes, size_t length)
    void CGContextShowGlyphs(CGContextRef context, CGGlyph glyphs[], size_t count)
    void CGContextShowTextAtPoint(CGContextRef context, float x, float y, 
        char* bytes, size_t length)
    void CGContextShowGlyphsAtPoint(CGContextRef context, float x, float y, 
        CGGlyph g[], size_t count)
    void CGContextShowGlyphsWithAdvances(CGContextRef c, CGGlyph glyphs[],
        CGSize advances[], size_t count)

    # Quartz Data Providers
    ctypedef void* CGDataProviderRef
    CGDataProviderRef CGDataProviderCreateWithData(void* info, void* data, 
        size_t size, void* callback)
    CGDataProviderRef CGDataProviderRetain(CGDataProviderRef provider)
    void CGDataProviderRelease(CGDataProviderRef provider)
    CGDataProviderRef CGDataProviderCreateWithURL(CFURLRef url)

    # Using Geometric Primitives
    CGPoint CGPointMake(float x, float y)
    CGSize CGSizeMake(float width, float height)
    CGRect CGRectMake(float x, float y, float width, float height)
    CGRect CGRectStandardize(CGRect rect)
    CGRect CGRectInset(CGRect rect, float dx, float dy)
    CGRect CGRectOffset(CGRect rect, float dx, float dy)
    CGRect CGRectIntegral(CGRect rect)
    CGRect CGRectUnion(CGRect r1, CGRect r2)
    CGRect CGRectIntersection(CGRect rect1, CGRect rect2)
    void CGRectDivide(CGRect rect, CGRect * slice, CGRect * remainder, 
        float amount, CGRectEdge edge)

    # Getting Geometric Information
    float CGRectGetMinX(CGRect rect)
    float CGRectGetMidX(CGRect rect)
    float CGRectGetMaxX(CGRect rect)
    float CGRectGetMinY(CGRect rect)
    float CGRectGetMidY(CGRect rect)
    float CGRectGetMaxY(CGRect rect)
    float CGRectGetWidth(CGRect rect)
    float CGRectGetHeight(CGRect rect)
    int CGRectIsNull(CGRect rect)
    int CGRectIsEmpty(CGRect rect)
    int CGRectIntersectsRect(CGRect rect1, CGRect rect2)
    int CGRectContainsRect(CGRect rect1, CGRect rect2)
    int CGRectContainsPoint(CGRect rect, CGPoint point)
    int CGRectEqualToRect(CGRect rect1, CGRect rect2)
    int CGSizeEqualToSize(CGSize size1, CGSize size2)
    int CGPointEqualToPoint(CGPoint point1, CGPoint point2)

    # Affine Transformations


    CGAffineTransform CGAffineTransformIdentity

    CGAffineTransform CGAffineTransformMake(float a, float b, float c, float d, 
        float tx, float ty)
    CGAffineTransform CGAffineTransformMakeTranslation(float tx, float ty)
    CGAffineTransform CGAffineTransformMakeScale(float sx, float sy)
    CGAffineTransform CGAffineTransformMakeRotation(float angle)
    CGAffineTransform CGAffineTransformTranslate(CGAffineTransform t, float tx, 
        float ty)
    CGAffineTransform CGAffineTransformScale(CGAffineTransform t, float sx, float sy)
    CGAffineTransform CGAffineTransformRotate(CGAffineTransform t, float angle)
    CGAffineTransform CGAffineTransformInvert(CGAffineTransform t)
    CGAffineTransform CGAffineTransformConcat(CGAffineTransform t1, 
        CGAffineTransform t2)
    CGPoint CGPointApplyAffineTransform(CGPoint point, CGAffineTransform t)
    CGSize CGSizeApplyAffineTransform(CGSize size, CGAffineTransform t)

    # Drawing Quartz Images
    ctypedef void*  CGImageRef

    ctypedef enum CGImageAlphaInfo:
        kCGImageAlphaNone
        kCGImageAlphaPremultipliedLast
        kCGImageAlphaPremultipliedFirst
        kCGImageAlphaLast
        kCGImageAlphaFirst
        kCGImageAlphaNoneSkipLast
        kCGImageAlphaNoneSkipFirst
        kCGImageAlphaOnly

    ctypedef enum CGInterpolationQuality:
        kCGInterpolationDefault
        kCGInterpolationNone
        kCGInterpolationLow
        kCGInterpolationHigh

    CGImageRef CGImageCreate(size_t width, size_t height, size_t bitsPerComponent, 
        size_t bitsPerPixel, size_t bytesPerRow, CGColorSpaceRef colorspace, 
        CGImageAlphaInfo alphaInfo, CGDataProviderRef provider, float decode[], 
        bool shouldInterpolate, CGColorRenderingIntent intent)
    CGImageRef CGImageMaskCreate(size_t width, size_t height, size_t bitsPerComponent,
        size_t bitsPerPixel, size_t bytesPerRow, CGDataProviderRef provider, 
        float decode[], bool shouldInterpolate)
    CGImageRef CGImageCreateWithJPEGDataProvider(CGDataProviderRef source, 
        float decode[], bool shouldInterpolate, CGColorRenderingIntent intent)
    CGImageRef CGImageCreateWithPNGDataProvider(CGDataProviderRef source, 
        float decode[], bool shouldInterpolate, CGColorRenderingIntent intent)
    CGImageRef CGImageCreateCopyWithColorSpace(CGImageRef image, 
        CGColorSpaceRef colorspace)
    CGImageRef CGImageRetain(CGImageRef image)
    void CGImageRelease(CGImageRef image)
    bool CGImageIsMask(CGImageRef image)
    size_t CGImageGetWidth(CGImageRef image)
    size_t CGImageGetHeight(CGImageRef image)
    size_t CGImageGetBitsPerComponent(CGImageRef image)
    size_t CGImageGetBitsPerPixel(CGImageRef image)
    size_t CGImageGetBytesPerRow(CGImageRef image)
    CGColorSpaceRef CGImageGetColorSpace(CGImageRef image)
    CGImageAlphaInfo CGImageGetAlphaInfo(CGImageRef image)
    CGDataProviderRef CGImageGetDataProvider(CGImageRef image)
    float *CGImageGetDecode(CGImageRef image)
    bool CGImageGetShouldInterpolate(CGImageRef image)
    CGColorRenderingIntent CGImageGetRenderingIntent(CGImageRef image)
    void CGContextDrawImage(CGContextRef context, CGRect rect, CGImageRef image)
    void CGContextSetInterpolationQuality(CGContextRef context, 
        CGInterpolationQuality quality)

    # PDF
    ctypedef void* CGPDFDocumentRef

    CGPDFDocumentRef CGPDFDocumentCreateWithProvider(CGDataProviderRef provider)
    CGPDFDocumentRef CGPDFDocumentCreateWithURL(CFURLRef url)
    void CGPDFDocumentRelease(CGPDFDocumentRef document)
    bool CGPDFDocumentUnlockWithPassword(CGPDFDocumentRef document, char *password)
    void CGContextDrawPDFDocument(CGContextRef context, CGRect rect, 
        CGPDFDocumentRef document, int page)

    size_t CGPDFDocumentGetNumberOfPages(CGPDFDocumentRef document)
    CGRect CGPDFDocumentGetMediaBox(CGPDFDocumentRef document, int page)
    CGRect CGPDFDocumentGetCropBox(CGPDFDocumentRef document, int page)
    CGRect CGPDFDocumentGetBleedBox(CGPDFDocumentRef document, int page)
    CGRect CGPDFDocumentGetTrimBox(CGPDFDocumentRef document, int page)
    CGRect CGPDFDocumentGetArtBox(CGPDFDocumentRef document, int page)
    int CGPDFDocumentGetRotationAngle(CGPDFDocumentRef document, int page)
    bool CGPDFDocumentAllowsCopying(CGPDFDocumentRef document)
    bool CGPDFDocumentAllowsPrinting(CGPDFDocumentRef document)
    bool CGPDFDocumentIsEncrypted(CGPDFDocumentRef document)
    bool CGPDFDocumentIsUnlocked(CGPDFDocumentRef document)


    # Bitmap Contexts
    CGContextRef CGBitmapContextCreate(void * data, size_t width, size_t height, 
        size_t bitsPerComponent, size_t bytesPerRow, CGColorSpaceRef colorspace, 
        CGImageAlphaInfo alphaInfo)
    CGImageAlphaInfo CGBitmapContextGetAlphaInfo(CGContextRef context)
    size_t CGBitmapContextGetBitsPerComponent(CGContextRef context)
    size_t CGBitmapContextGetBitsPerPixel(CGContextRef context)
    size_t CGBitmapContextGetBytesPerRow(CGContextRef context)
    CGColorSpaceRef CGBitmapContextGetColorSpace(CGContextRef context)
    void *CGBitmapContextGetData(CGContextRef context)
    size_t CGBitmapContextGetHeight(CGContextRef context)
    size_t CGBitmapContextGetWidth(CGContextRef context)

    # Path Handling
    ctypedef void* CGMutablePathRef
    ctypedef void* CGPathRef

    CGMutablePathRef CGPathCreateMutable()
    CGPathRef CGPathCreateCopy(CGPathRef path)
    CGMutablePathRef CGPathCreateMutableCopy(CGPathRef path)
    CGPathRef CGPathRetain(CGPathRef path)
    void CGPathRelease(CGPathRef path)
    bool CGPathEqualToPath(CGPathRef path1, CGPathRef path2)
    void CGPathMoveToPoint(CGMutablePathRef path, CGAffineTransform *m, float x, float y)
    void CGPathAddLineToPoint(CGMutablePathRef path, CGAffineTransform *m, float x, float y)
    void CGPathAddQuadCurveToPoint(CGMutablePathRef path, CGAffineTransform *m,
        float cpx, float cpy, float x, float y)
    void CGPathAddCurveToPoint(CGMutablePathRef path, CGAffineTransform *m,
        float cp1x, float cp1y, float cp2x, float cp2y, float x, float y)
    void CGPathCloseSubpath(CGMutablePathRef path)
    void CGPathAddRect(CGMutablePathRef path, CGAffineTransform *m, CGRect rect)
    void CGPathAddRects(CGMutablePathRef path, CGAffineTransform *m, 
        CGRect rects[], size_t count)
    void CGPathAddLines(CGMutablePathRef path, CGAffineTransform *m, 
        CGPoint points[], size_t count)
    void CGPathAddArc(CGMutablePathRef path, CGAffineTransform *m, 
        float x, float y, float radius, float startAngle, float endAngle, bool clockwise)
    void CGPathAddArcToPoint(CGMutablePathRef path, CGAffineTransform *m,
        float x1, float y1, float x2, float y2, float radius)
    void CGPathAddPath(CGMutablePathRef path1, CGAffineTransform *m, CGPathRef path2)
    bool CGPathIsEmpty(CGPathRef path)
    bool CGPathIsRect(CGPathRef path, CGRect *rect)
    CGPoint CGPathGetCurrentPoint(CGPathRef path)
    CGRect CGPathGetBoundingBox(CGPathRef path)
    void CGContextAddPath(CGContextRef context, CGPathRef path)


    ctypedef enum CGPathElementType:
        kCGPathElementMoveToPoint,
        kCGPathElementAddLineToPoint,
        kCGPathElementAddQuadCurveToPoint,
        kCGPathElementAddCurveToPoint,
        kCGPathElementCloseSubpath

    ctypedef struct CGPathElement:
        CGPathElementType type
        CGPoint *points

    ctypedef void (*CGPathApplierFunction)(void *info, CGPathElement *element)

    void CGPathApply(CGPathRef path, void *info, CGPathApplierFunction function)

    CGContextRef CGGLContextCreate(void *glContext, CGSize size, 
        CGColorSpaceRef colorspace)
    void CGGLContextUpdateViewportSize(CGContextRef context, CGSize size)

    
    ctypedef void* CGFunctionRef

    ctypedef void (*CGFunctionEvaluateCallback)(void *info, float *in_data, float *out)

    ctypedef void (*CGFunctionReleaseInfoCallback)(void *info)

    ctypedef struct CGFunctionCallbacks:
        unsigned int version
        CGFunctionEvaluateCallback evaluate
        CGFunctionReleaseInfoCallback releaseInfo
    
    CGFunctionRef CGFunctionCreate(void *info, size_t domainDimension, 
        float *domain, size_t rangeDimension, float *range, 
        CGFunctionCallbacks *callbacks)
    CGFunctionRef CGFunctionRetain(CGFunctionRef function)
    void CGFunctionRelease(CGFunctionRef function)

    ctypedef void* CGShadingRef

    CGShadingRef CGShadingCreateAxial(CGColorSpaceRef colorspace, CGPoint start,
        CGPoint end, CGFunctionRef function, bool extendStart, bool extendEnd)
    CGShadingRef CGShadingCreateRadial(CGColorSpaceRef colorspace, 
        CGPoint start, float startRadius, CGPoint end, float endRadius, 
        CGFunctionRef function, bool extendStart, bool extendEnd)
    CGShadingRef CGShadingRetain(CGShadingRef shading)
    void CGShadingRelease(CGShadingRef shading)

    void CGContextDrawShading(CGContextRef context, CGShadingRef shading)

    # Transparency Layers
    void CGContextBeginTransparencyLayer(CGContextRef context, CFDictionaryRef auxiliaryInfo)
    void CGContextEndTransparencyLayer(CGContextRef context)

    # CGLayers
    ctypedef void* CGLayerRef
    CGLayerRef CGLayerCreateWithContext(CGContextRef context, CGSize size, CFDictionaryRef auxiliaryInfo)
    CGLayerRef CGLayerRetain(CGLayerRef layer)
    void CGLayerRelease(CGLayerRef layer)
    CGSize CGLayerGetSize(CGLayerRef layer)
    CGContextRef CGLayerGetContext(CGLayerRef layer)
    void CGContextDrawLayerInRect(CGContextRef context, CGRect rect, CGLayerRef layer)
    void CGContextDrawLayerAtPoint(CGContextRef context, CGPoint point, CGLayerRef layer)
    CFTypeID CGLayerGetTypeID()

    CFTypeID CGContextGetTypeID()

cdef CGRect CGRectMakeFromPython(object seq):
    return CGRectMake(seq[0], seq[1], seq[2], seq[3])
