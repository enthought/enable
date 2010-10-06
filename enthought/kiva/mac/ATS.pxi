# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style


cdef extern from "ATSFont.h":

    ctypedef struct Float32Point:
        float x
        float y

    ctypedef unsigned long UInt32
    ctypedef short SInt16

    ctypedef UInt32                          FMGeneration
    ctypedef SInt16                          FMFontFamily
    ctypedef SInt16                          FMFontStyle
    ctypedef SInt16                          FMFontSize
    ctypedef UInt32                          FMFont

    ctypedef struct FMFontFamilyInstance:
      FMFontFamily        fontFamily
      FMFontStyle         fontStyle

    ctypedef struct FMFontFamilyIterator:
      UInt32              reserved[16]

    ctypedef struct FMFontIterator:
      UInt32              reserved[16]

    ctypedef struct FMFontFamilyInstanceIterator:
      UInt32              reserved[16]

    cdef enum:
      kInvalidGeneration            = 0
      kInvalidFontFamily            = -1
      kInvalidFont                  = 0

    cdef enum:
      kFMCurrentFilterFormat        = 0

    ctypedef UInt32 FMFilterSelector
    cdef enum:
      kFMFontTechnologyFilterSelector = 1
      kFMFontContainerFilterSelector = 2
      kFMGenerationFilterSelector   = 3
      kFMFontFamilyCallbackFilterSelector = 4
      kFMFontCallbackFilterSelector = 5
      kFMFontDirectoryFilterSelector = 6

    #kFMTrueTypeFontTechnology     = 'true'
    #kFMPostScriptFontTechnology   = 'typ1'

    #struct FMFontDirectoryFilter:
    #  SInt16              fontFolderDomain
    #  UInt32              reserved[2]
    #
    #typedef struct FMFontDirectoryFilter    FMFontDirectoryFilter
    #struct FMFilter:
    #  UInt32              format
    #  FMFilterSelector    selector
    #  union:
    #    FourCharCode        fontTechnologyFilter
    #    FSSpec              fontContainerFilter
    #    FMGeneration        generationFilter
    #    FMFontFamilyCallbackFilterUPP  fontFamilyCallbackFilter
    #    FMFontCallbackFilterUPP  fontCallbackFilter
    #    FMFontDirectoryFilter  fontDirectoryFilter
    #  }                       filter
    #
    #typedef struct FMFilter                 FMFilter
    #
    ctypedef float Float32
    ctypedef unsigned long OptionBits
    ctypedef unsigned short UInt16

    ctypedef OptionBits                      ATSOptionFlags
    ctypedef UInt32                          ATSGeneration
    ctypedef UInt32                          ATSFontContainerRef
    ctypedef UInt32                          ATSFontFamilyRef
    ctypedef UInt32                          ATSFontRef
    ctypedef UInt16                          ATSGlyphRef
    ctypedef Float32                         ATSFontSize

    cdef enum:
      kATSGenerationUnspecified     = 0
      kATSFontContainerRefUnspecified = 0
      kATSFontFamilyRefUnspecified  = 0
      kATSFontRefUnspecified        = 0


    ctypedef struct ATSFontMetrics:
      UInt32              version
      Float32             ascent
      Float32             descent
      Float32             leading
      Float32             avgAdvanceWidth
      Float32             maxAdvanceWidth
      Float32             minLeftSideBearing
      Float32             minRightSideBearing
      Float32             stemWidth
      Float32             stemHeight
      Float32             capHeight
      Float32             xHeight
      Float32             italicAngle
      Float32             underlinePosition
      Float32             underlineThickness

    cdef enum:
      kATSItalicQDSkew              = 16384  # (1 << 16) / 4
      kATSBoldQDStretch             = 98304  # (1 << 16) * 3 / 2
      kATSRadiansFactor             = 1144


    ctypedef UInt16 ATSCurveType
    cdef enum:
      kATSCubicCurveType            = 0x0001
      kATSQuadCurveType             = 0x0002
      kATSOtherCurveType            = 0x0003


    cdef enum:
      kATSDeletedGlyphcode          = 0xFFFF

    ctypedef struct ATSUCurvePath:
      UInt32              vectors
      UInt32              controlBits[1]
      Float32Point        vector[1]

    ctypedef struct ATSUCurvePaths:
      UInt32              contours
      ATSUCurvePath       contour[1]

    ctypedef struct ATSGlyphIdealMetrics:
      Float32Point        advance
      Float32Point        sideBearing
      Float32Point        otherSideBearing

    ctypedef struct ATSGlyphScreenMetrics:
      Float32Point        deviceAdvance
      Float32Point        topLeft
      UInt32              height
      UInt32              width
      Float32Point        sideBearing
      Float32Point        otherSideBearing

    ctypedef ATSGlyphRef                     GlyphID

    cdef enum:
      kATSOptionFlagsDefault        = 0
      kATSOptionFlagsComposeFontPostScriptName = 1 << 0
      kATSOptionFlagsUseDataForkAsResourceFork = 1 << 8
      kATSOptionFlagsUseResourceFork = 2 << 8
      kATSOptionFlagsUseDataFork    = 3 << 8

    cdef enum:
      kATSIterationCompleted        = -980
      kATSInvalidFontFamilyAccess   = -981
      kATSInvalidFontAccess         = -982
      kATSIterationScopeModified    = -983
      kATSInvalidFontTableAccess    = -984
      kATSInvalidFontContainerAccess = -985
      kATSInvalidGlyphAccess        = -986

    ctypedef long ATSFontContext

    cdef enum:
      kATSFontContextUnspecified    = 0
      kATSFontContextGlobal         = 1
      kATSFontContextLocal          = 2

    cdef enum:
      kATSOptionFlagsProcessSubdirectories = 0x00000001 << 6
      kATSOptionFlagsDoNotNotify    = 0x00000001 << 7

    cdef enum:
      kATSOptionFlagsIterateByPrecedenceMask = 0x00000001 << 5
      kATSOptionFlagsIterationScopeMask = 0x00000007 << 12
      kATSOptionFlagsDefaultScope   = 0x00000000 << 12
      kATSOptionFlagsUnRestrictedScope = 0x00000001 << 12
      kATSOptionFlagsRestrictedScope = 0x00000002 << 12

    ctypedef long ATSFontFormat

    cdef enum:
      kATSFontFormatUnspecified     = 0

    ctypedef OSStatus (*ATSFontFamilyApplierFunction)(ATSFontFamilyRef iFamily,
        void *iRefCon)
    ctypedef OSStatus (*ATSFontApplierFunction)(ATSFontRef iFont, void *iRefCon)
    ctypedef void* ATSFontFamilyIterator
    ctypedef void* ATSFontIterator

    cdef enum:
      kATSFontFilterCurrentVersion  = 0

    ctypedef enum ATSFontFilterSelector:
      kATSFontFilterSelectorUnspecified = 0
      kATSFontFilterSelectorGeneration = 3
      kATSFontFilterSelectorFontFamily = 7
      kATSFontFilterSelectorFontFamilyApplierFunction = 8
      kATSFontFilterSelectorFontApplierFunction = 9

    ctypedef union font_filter:
        ATSGeneration       generationFilter
        ATSFontFamilyRef    fontFamilyFilter
        ATSFontFamilyApplierFunction  fontFamilyApplierFunctionFilter
        ATSFontApplierFunction  fontApplierFunctionFilter

    ctypedef struct ATSFontFilter:
        UInt32 version
        ATSFontFilterSelector  filterSelector

        font_filter filter

    ctypedef void*  ATSFontNotificationRef
    ctypedef void*  ATSFontNotificationInfoRef

    ctypedef enum ATSFontNotifyOption:
      kATSFontNotifyOptionDefault   = 0
      kATSFontNotifyOptionReceiveWhileSuspended = 1 << 0


    ctypedef enum ATSFontNotifyAction:
      kATSFontNotifyActionFontsChanged = 1
      kATSFontNotifyActionDirectoriesChanged = 2

    ATSGeneration  ATSGetGeneration()
    OSStatus ATSFontFamilyApplyFunction(ATSFontFamilyApplierFunction iFunction,
        void* iRefCon)


    OSStatus ATSFontFamilyIteratorCreate(
      ATSFontContext           iContext,
      ATSFontFilter *    iFilter,
      void *                   iRefCon,
      ATSOptionFlags           iOptions,
      ATSFontFamilyIterator *  ioIterator)

    OSStatus ATSFontFamilyIteratorRelease(ATSFontFamilyIterator * ioIterator)


    OSStatus ATSFontFamilyIteratorReset(
      ATSFontContext           iContext,
      ATSFontFilter *    iFilter,
      void *                   iRefCon,
      ATSOptionFlags           iOptions,
      ATSFontFamilyIterator *  ioIterator)

    OSStatus ATSFontFamilyIteratorNext(
      ATSFontFamilyIterator   iIterator,
      ATSFontFamilyRef *      oFamily)

    ATSFontFamilyRef ATSFontFamilyFindFromName(
      CFStringRef      iName,
      ATSOptionFlags   iOptions)

    ATSGeneration ATSFontFamilyGetGeneration(ATSFontFamilyRef iFamily)

    OSStatus ATSFontFamilyGetName(
      ATSFontFamilyRef   iFamily,
      ATSOptionFlags     iOptions,
      CFStringRef *      oName)

    ctypedef UInt32 TextEncoding

    TextEncoding ATSFontFamilyGetEncoding(ATSFontFamilyRef iFamily)

    OSStatus ATSFontApplyFunction(ATSFontApplierFunction iFunction,
        void* iRefCon)


    OSStatus ATSFontIteratorCreate(
      ATSFontContext         iContext,
      ATSFontFilter *  iFilter,
      void *                 iRefCon,
      ATSOptionFlags         iOptions,
      ATSFontIterator *      ioIterator)

    OSStatus ATSFontIteratorRelease(ATSFontIterator * ioIterator)

    OSStatus ATSFontIteratorReset(
      ATSFontContext         iContext,
      ATSFontFilter *  iFilter,
      void *                 iRefCon,
      ATSOptionFlags         iOptions,
      ATSFontIterator *      ioIterator)

    OSStatus ATSFontIteratorNext(
      ATSFontIterator   iIterator,
      ATSFontRef *      oFont)

    ATSFontRef ATSFontFindFromName(
      CFStringRef      iName,
      ATSOptionFlags   iOptions)

    ATSFontRef ATSFontFindFromPostScriptName(
      CFStringRef      iName,
      ATSOptionFlags   iOptions)

#    OSStatus ATSFontFindFromContainer(
#      ATSFontContainerRef   iContainer,
#      ATSOptionFlags        iOptions,
#      ItemCount             iCount,
#      ATSFontRef            ioArray[],
#      ItemCount *           oCount)
#
    ATSGeneration ATSFontGetGeneration(ATSFontRef iFont)

    OSStatus ATSFontGetName(
      ATSFontRef       iFont,
      ATSOptionFlags   iOptions,
      CFStringRef *    oName)

    OSStatus ATSFontGetPostScriptName(
      ATSFontRef       iFont,
      ATSOptionFlags   iOptions,
      CFStringRef *    oName)

#    OSStatus ATSFontGetTableDirectory(
#      ATSFontRef   iFont,
#      ByteCount    iBufferSize,
#      void *       ioBuffer,
#      ByteCount *  oSize)

#    ctypedef char[4] FourCharCode
#    OSStatus ATSFontGetTable(
#      ATSFontRef     iFont,
#      FourCharCode   iTag,
#      ByteOffset     iOffset,
#      ByteCount      iBufferSize,
#      void *         ioBuffer,
#      ByteCount *    oSize)

    OSStatus ATSFontGetHorizontalMetrics(
      ATSFontRef        iFont,
      ATSOptionFlags    iOptions,
      ATSFontMetrics *  oMetrics)

    OSStatus ATSFontGetVerticalMetrics(
      ATSFontRef        iFont,
      ATSOptionFlags    iOptions,
      ATSFontMetrics *  oMetrics)

    ctypedef char ConstStr255Param[256]
    ATSFontFamilyRef ATSFontFamilyFindFromQuickDrawName(ConstStr255Param iName)

    ctypedef char Str255[256]
    OSStatus ATSFontFamilyGetQuickDrawName(
      ATSFontFamilyRef   iFamily,
      Str255             oName)

#    OSStatus ATSFontGetFileSpecification(
#      ATSFontRef   iFont,
#      FSSpec *     oFile)

#    OSStatus ATSFontGetFontFamilyResource(
#      ATSFontRef   iFont,
#      ByteCount    iBufferSize,
#      void *       ioBuffer,
#      ByteCount *  oSize)
#

    #ctypedef struct ATSFontQuerySourceContext:
    #  UInt32              version
    #  void *              refCon
    #  CFAllocatorRetainCallBack  retain
    #  CFAllocatorReleaseCallBack  release
    #

    #ctypedef enum ATSFontQueryMessageID:
    #  kATSQueryActivateFontMessage  = 'atsa'

cdef extern from "Fonts.h":

    OSStatus FMGetFontFamilyInstanceFromFont(FMFont iFont, FMFontFamily* oFontFamily,
        FMFontStyle* oStyle)
    ATSFontRef FMGetATSFontRefFromFont(FMFont iFont)
    FMFont FMGetFontFromATSFontRef(ATSFontRef iFont)


    cdef enum:
        normal                        = 0
        bold                          = 1
        italic                        = 2
        underline                     = 4
        outline                       = 8
        shadow                        = 0x10
        condense                      = 0x20
        extend                        = 0x40  

