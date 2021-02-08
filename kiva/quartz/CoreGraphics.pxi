# (C) Copyright 2004-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

cdef extern from "ApplicationServices/ApplicationServices.h":
    ctypedef void*  CGContextRef

    ctypedef double CGFloat

    ctypedef struct CGPoint:
        CGFloat x
        CGFloat y

    ctypedef struct CGSize:
        CGFloat width
        CGFloat height

    ctypedef struct CGRect:
        CGPoint origin
        CGSize size

    ctypedef struct CGAffineTransform:
        CGFloat a
        CGFloat b
        CGFloat c
        CGFloat d
        CGFloat tx
        CGFloat ty

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
    void CGContextScaleCTM(CGContextRef context, CGFloat sx, CGFloat sy)
    void CGContextTranslateCTM(CGContextRef context, CGFloat tx, CGFloat ty)
    void CGContextRotateCTM(CGContextRef context, CGFloat angle)
    void CGContextConcatCTM(CGContextRef context, CGAffineTransform transform)
    CGAffineTransform CGContextGetCTM(CGContextRef context)

    # Saving and Restoring the Current Graphics State
    void CGContextSaveGState(CGContextRef context)
    void CGContextRestoreGState(CGContextRef context)

    # Changing the Current Graphics State
    void CGContextSetFlatness(CGContextRef context, CGFloat flatness)
    void CGContextSetLineCap(CGContextRef context, CGLineCap cap)
    void CGContextSetLineDash(CGContextRef context, CGFloat phase,
        CGFloat lengths[], size_t count)
    void CGContextSetLineJoin(CGContextRef context, CGLineJoin join)
    void CGContextSetLineWidth(CGContextRef context, CGFloat width)
    void CGContextSetMiterLimit(CGContextRef context, CGFloat limit)
    void CGContextSetPatternPhase(CGContextRef context, CGSize phase)
    void CGContextSetShouldAntialias(CGContextRef context, bool shouldAntialias)
    void CGContextSetShouldSmoothFonts(CGContextRef context, bool shouldSmoothFonts)
    void CGContextSetAllowsAntialiasing(CGContextRef context, bool allowsAntialiasing)

    # Constructing Paths
    void CGContextBeginPath(CGContextRef context)
    void CGContextMoveToPoint(CGContextRef context, CGFloat x, CGFloat y)
    void CGContextAddLineToPoint(CGContextRef context, CGFloat x, CGFloat y)
    void CGContextAddLines(CGContextRef context, CGPoint points[], size_t count)
    void CGContextAddCurveToPoint(CGContextRef context, CGFloat cp1x, CGFloat cp1y,
        CGFloat cp2x, CGFloat cp2y, CGFloat x, CGFloat y)
    void CGContextAddQuadCurveToPoint(CGContextRef context, CGFloat cpx,
        CGFloat cpy, CGFloat x, CGFloat y)
    void CGContextAddRect(CGContextRef context, CGRect rect)
    void CGContextAddRects(CGContextRef context, CGRect rects[], size_t count)
    void CGContextAddArc(CGContextRef context, CGFloat x, CGFloat y, CGFloat radius,
        CGFloat startAngle, CGFloat endAngle, int clockwise)
    void CGContextAddArcToPoint(CGContextRef context, CGFloat x1, CGFloat y1,
        CGFloat x2, CGFloat y2, CGFloat radius)
    void CGContextClosePath(CGContextRef context)
    void CGContextAddEllipseInRect(CGContextRef context, CGRect rect)

    # Painting Paths
    void CGContextDrawPath(CGContextRef context, CGPathDrawingMode mode)
    void CGContextStrokePath(CGContextRef context)
    void CGContextFillPath(CGContextRef context)
    void CGContextEOFillPath(CGContextRef context)
    void CGContextStrokeRect(CGContextRef context, CGRect rect)
    void CGContextStrokeRectWithWidth(CGContextRef context, CGRect rect, CGFloat width)
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
    CGRect CGContextGetClipBoundingBox(CGContextRef context)

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
    CGColorRef CGColorCreate(CGColorSpaceRef colorspace, CGFloat components[])
#    CGColorRef CGColorCreateWithPattern(CGColorSpaceRef colorspace, CGPatternRef pattern, CGFloat components[])
    CGColorRef CGColorCreateCopy(CGColorRef color)
    CGColorRef CGColorCreateCopyWithAlpha(CGColorRef color, CGFloat alpha)
    CGColorRef CGColorCreateGenericRGB(CGFloat red, CGFloat green, CGFloat blue, CGFloat alpha)
    CGColorRef CGColorRetain(CGColorRef color)
    void CGColorRelease(CGColorRef color)
    bool CGColorEqualToColor(CGColorRef color1, CGColorRef color2)
    size_t CGColorGetNumberOfComponents(CGColorRef color)
    CGFloat *CGColorGetComponents(CGColorRef color)
    CGFloat CGColorGetAlpha(CGColorRef color)
    CGColorSpaceRef CGColorGetColorSpace(CGColorRef color)
#    CGPatternRef CGColorGetPattern(CGColorRef color)

    void CGContextSetFillColorSpace(CGContextRef context, CGColorSpaceRef colorspace)
    void CGContextSetFillColor(CGContextRef context, CGFloat components[])
    void CGContextSetStrokeColorSpace(CGContextRef context,
        CGColorSpaceRef colorspace)
    void CGContextSetStrokeColor(CGContextRef context, CGFloat components[])
    void CGContextSetGrayFillColor(CGContextRef context, CGFloat gray, CGFloat alpha)
    void CGContextSetGrayStrokeColor(CGContextRef context, CGFloat gray, CGFloat alpha)
    void CGContextSetRGBFillColor(CGContextRef context, CGFloat red, CGFloat green,
        CGFloat blue, CGFloat alpha)
    void CGContextSetRGBStrokeColor(CGContextRef context, CGFloat red, CGFloat green,
        CGFloat blue, CGFloat alpha)
    void CGContextSetCMYKFillColor(CGContextRef context, CGFloat cyan, CGFloat magenta,
        CGFloat yellow, CGFloat black, CGFloat alpha)
    void CGContextSetCMYKStrokeColor(CGContextRef context, CGFloat cyan, CGFloat magenta,
        CGFloat yellow, CGFloat black, CGFloat alpha)
    void CGContextSetAlpha(CGContextRef context, CGFloat alpha)
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

    void CGContextSelectFont(CGContextRef context, char* name, CGFloat size,
        CGTextEncoding textEncoding)
    void CGContextSetFontSize(CGContextRef context, CGFloat size)
    void CGContextSetCharacterSpacing(CGContextRef context, CGFloat spacing)
    void CGContextSetTextDrawingMode(CGContextRef context, CGTextDrawingMode mode)
    void CGContextSetTextPosition(CGContextRef context, CGFloat x, CGFloat y)
    CGPoint CGContextGetTextPosition(CGContextRef context)
    void CGContextSetTextMatrix(CGContextRef context, CGAffineTransform transform)
    CGAffineTransform CGContextGetTextMatrix(CGContextRef context)
    void CGContextShowText(CGContextRef context, char* bytes, size_t length)
    void CGContextShowGlyphs(CGContextRef context, CGGlyph glyphs[], size_t count)
    void CGContextShowTextAtPoint(CGContextRef context, CGFloat x, CGFloat y,
        char* bytes, size_t length)
    void CGContextShowGlyphsAtPoint(CGContextRef context, CGFloat x, CGFloat y,
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
    CGPoint CGPointMake(CGFloat x, CGFloat y)
    CGSize CGSizeMake(CGFloat width, CGFloat height)
    CGRect CGRectMake(CGFloat x, CGFloat y, CGFloat width, CGFloat height)
    CGRect CGRectStandardize(CGRect rect)
    CGRect CGRectInset(CGRect rect, CGFloat dx, CGFloat dy)
    CGRect CGRectOffset(CGRect rect, CGFloat dx, CGFloat dy)
    CGRect CGRectIntegral(CGRect rect)
    CGRect CGRectUnion(CGRect r1, CGRect r2)
    CGRect CGRectIntersection(CGRect rect1, CGRect rect2)
    void CGRectDivide(CGRect rect, CGRect * slice, CGRect * remainder,
        CGFloat amount, CGRectEdge edge)

    # Getting Geometric Information
    CGFloat CGRectGetMinX(CGRect rect)
    CGFloat CGRectGetMidX(CGRect rect)
    CGFloat CGRectGetMaxX(CGRect rect)
    CGFloat CGRectGetMinY(CGRect rect)
    CGFloat CGRectGetMidY(CGRect rect)
    CGFloat CGRectGetMaxY(CGRect rect)
    CGFloat CGRectGetWidth(CGRect rect)
    CGFloat CGRectGetHeight(CGRect rect)
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

    CGAffineTransform CGAffineTransformMake(CGFloat a, CGFloat b, CGFloat c, CGFloat d,
        CGFloat tx, CGFloat ty)
    CGAffineTransform CGAffineTransformMakeTranslation(CGFloat tx, CGFloat ty)
    CGAffineTransform CGAffineTransformMakeScale(CGFloat sx, CGFloat sy)
    CGAffineTransform CGAffineTransformMakeRotation(CGFloat angle)
    CGAffineTransform CGAffineTransformTranslate(CGAffineTransform t, CGFloat tx,
        CGFloat ty)
    CGAffineTransform CGAffineTransformScale(CGAffineTransform t, CGFloat sx, CGFloat sy)
    CGAffineTransform CGAffineTransformRotate(CGAffineTransform t, CGFloat angle)
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
        CGImageAlphaInfo alphaInfo, CGDataProviderRef provider, CGFloat decode[],
        bool shouldInterpolate, CGColorRenderingIntent intent)
    CGImageRef CGImageMaskCreate(size_t width, size_t height, size_t bitsPerComponent,
        size_t bitsPerPixel, size_t bytesPerRow, CGDataProviderRef provider,
        CGFloat decode[], bool shouldInterpolate)
    CGImageRef CGImageCreateWithJPEGDataProvider(CGDataProviderRef source,
        CGFloat decode[], bool shouldInterpolate, CGColorRenderingIntent intent)
    CGImageRef CGImageCreateWithPNGDataProvider(CGDataProviderRef source,
        CGFloat decode[], bool shouldInterpolate, CGColorRenderingIntent intent)
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
    CGFloat *CGImageGetDecode(CGImageRef image)
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
    void CGPathMoveToPoint(CGMutablePathRef path, CGAffineTransform *m, CGFloat x, CGFloat y)
    void CGPathAddLineToPoint(CGMutablePathRef path, CGAffineTransform *m, CGFloat x, CGFloat y)
    void CGPathAddQuadCurveToPoint(CGMutablePathRef path, CGAffineTransform *m,
        CGFloat cpx, CGFloat cpy, CGFloat x, CGFloat y)
    void CGPathAddCurveToPoint(CGMutablePathRef path, CGAffineTransform *m,
        CGFloat cp1x, CGFloat cp1y, CGFloat cp2x, CGFloat cp2y, CGFloat x, CGFloat y)
    void CGPathCloseSubpath(CGMutablePathRef path)
    void CGPathAddRect(CGMutablePathRef path, CGAffineTransform *m, CGRect rect)
    void CGPathAddRects(CGMutablePathRef path, CGAffineTransform *m,
        CGRect rects[], size_t count)
    void CGPathAddLines(CGMutablePathRef path, CGAffineTransform *m,
        CGPoint points[], size_t count)
    void CGPathAddArc(CGMutablePathRef path, CGAffineTransform *m,
        CGFloat x, CGFloat y, CGFloat radius, CGFloat startAngle, CGFloat endAngle, bool clockwise)
    void CGPathAddArcToPoint(CGMutablePathRef path, CGAffineTransform *m,
        CGFloat x1, CGFloat y1, CGFloat x2, CGFloat y2, CGFloat radius)
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

    ctypedef void* CGFunctionRef

    ctypedef void (*CGFunctionEvaluateCallback)(void *info, CGFloat *in_data, CGFloat *out)

    ctypedef void (*CGFunctionReleaseInfoCallback)(void *info)

    ctypedef struct CGFunctionCallbacks:
        unsigned int version
        CGFunctionEvaluateCallback evaluate
        CGFunctionReleaseInfoCallback releaseInfo

    CGFunctionRef CGFunctionCreate(void *info, size_t domainDimension,
        CGFloat *domain, size_t rangeDimension, CGFloat *range,
        CGFunctionCallbacks *callbacks)
    CGFunctionRef CGFunctionRetain(CGFunctionRef function)
    void CGFunctionRelease(CGFunctionRef function)

    ctypedef void* CGShadingRef

    CGShadingRef CGShadingCreateAxial(CGColorSpaceRef colorspace, CGPoint start,
        CGPoint end, CGFunctionRef function, bool extendStart, bool extendEnd)
    CGShadingRef CGShadingCreateRadial(CGColorSpaceRef colorspace,
        CGPoint start, CGFloat startRadius, CGPoint end, CGFloat endRadius,
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

