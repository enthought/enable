# :Author:    Robert Kern
# :Copyright: 2006, Enthought, Inc.
# :License:   BSD Style


cdef extern from "ATSUnicode.h":

    #### Types #################################################################

    ctypedef unsigned int ATSUFontID
    ctypedef void* ATSUTextLayout
    ctypedef void* ATSUStyle
    ctypedef void* ATSUFontFallbacks
    ctypedef int Fixed
    Fixed FloatToFixed(float x)
    float FixedToFloat(Fixed x)
    ctypedef unsigned short ATSUFontFeatureType
    ctypedef unsigned short ATSUFontFeatureSelector
    ctypedef int ATSUAttributeTag

    ctypedef enum:
        kATSULineWidthTag
        kATSULineRotationTag
        kATSULineDirectionTag
        kATSULineJustificationFactorTag
        kATSULineFlushFactorTag
        kATSULineBaselineValuesTag
        kATSULineLayoutOptionsTag
        kATSULineAscentTag
        kATSULineDescentTag
        kATSULineLangRegionTag
        kATSULineTextLocatorTag
        kATSULineTruncationTag
        kATSULineFontFallbacksTag
        kATSULineDecimalTabCharacterTag
        kATSULayoutOperationOverrideTag
        kATSULineHighlightCGColorTag
        kATSUMaxLineTag
        kATSULineLanguageTag
        kATSUCGContextTag
        kATSUQDBoldfaceTag
        kATSUQDItalicTag
        kATSUQDUnderlineTag
        kATSUQDCondensedTag
        kATSUQDExtendedTag
        kATSUFontTag
        kATSUSizeTag
        kATSUColorTag
        kATSULangRegionTag
        kATSUVerticalCharacterTag
        kATSUImposeWidthTag
        kATSUBeforeWithStreamShiftTag
        kATSUAfterWithStreamShiftTag
        kATSUCrossStreamShiftTag
        kATSUTrackingTag
        kATSUHangingInhibitFactorTag
        kATSUKerningInhibitFactorTag
        kATSUDecompositionFactorTag
        kATSUBaselineClassTag
        kATSUPriorityJustOverrideTag
        kATSUNoLigatureSplitTag
        kATSUNoCaretAngleTag
        kATSUSuppressCrossKerningTag
        kATSUNoOpticalAlignmentTag
        kATSUForceHangingTag
        kATSUNoSpecialJustificationTag
        kATSUStyleTextLocatorTag
        kATSUStyleRenderingOptionsTag
        kATSUAscentTag
        kATSUDescentTag
        kATSULeadingTag
        kATSUGlyphSelectorTag
        kATSURGBAlphaColorTag
        kATSUStyleUnderlineCountOptionTag
        kATSUStyleUnderlineColorOptionTag
        kATSUStyleStrikeThroughTag
        kATSUStyleStrikeThroughCountOptionTag
        kATSUStyleStrikeThroughColorOptionTag
        kATSUStyleDropShadowTag
        kATSUStyleDropShadowBlurOptionTag
        kATSUStyleDropShadowOffsetOptionTag
        kATSUStyleDropShadowColorOptionTag
        kATSUMaxStyleTag
        kATSULanguageTag
        kATSUMaxATSUITagValue

    ctypedef enum:
        kFontCopyrightName
        kFontFamilyName
        kFontStyleName
        kFontUniqueName
        kFontFullName
        kFontVersionName
        kFontPostscriptName
        kFontTrademarkName
        kFontManufacturerName
        kFontDesignerName
        kFontDescriptionName
        kFontVendorURLName
        kFontDesignerURLName
        kFontLicenseDescriptionName
        kFontLicenseInfoURLName
        kFontLastReservedName

    ctypedef enum:
        kFontNoPlatformCode
        kFontNoScriptCode
        kFontNoLanguageCode

    ctypedef void* ATSUAttributeValuePtr
    ctypedef void* ConstATSUAttributeValuePtr

    ctypedef struct ATSURGBAlphaColor:
        float red
        float green
        float blue
        float alpha

    #ctypedef int OSStatus

    ctypedef enum:
        kATSULeftToRightBaseDirection = 0
        kATSURightToLeftBaseDirection = 1

    ctypedef enum:
        kATSUStartAlignment = 0x00000000
        kATSUEndAlignment = 0x40000000
        kATSUCenterAlignment = 0x20000000
        kATSUNoJustification = 0x00000000
        kATSUFullJustification = 0x40000000

    ctypedef enum:
        kATSUInvalidFontID = 0

    ctypedef enum:
        kATSUUseLineControlWidth = 0x7FFFFFFF

    ctypedef enum:
        kATSUNoSelector = 0x0000FFFF

    ctypedef enum:
        kATSUFromTextBeginning #= (unsigned long)0xFFFFFFFF
        kATSUToTextEnd #= (unsigned long)0xFFFFFFFF
        kATSUFromPreviousLayout #= (unsigned long)0xFFFFFFFE
        kATSUFromFollowingLayout #= (unsigned long)0xFFFFFFFD

    ctypedef unsigned short ATSUFontFallbackMethod
    ctypedef enum:
        kATSUDefaultFontFallbacks
        kATSULastResortOnlyFallback
        kATSUSequentialFallbacksPreferred
        kATSUSequentialFallbacksExclusive


    ctypedef unsigned int ByteCount
    ctypedef unsigned int ItemCount
    ctypedef unsigned int* ConstUniCharArrayPtr
    ctypedef unsigned int UniCharArrayOffset
    ctypedef unsigned int UniCharCount
    #ctypedef int Boolean
    ctypedef void* Ptr
    #ctypedef unsigned int UInt32
    #ctypedef unsigned short UInt16
    ctypedef UInt32 FontNameCode
    ctypedef UInt32 FontPlatformCode
    ctypedef UInt32 FontScriptCode
    ctypedef UInt32 FontLanguageCode
#    ctypedef struct QDRect "Rect":
#        short    top
#        short    left
#        short    bottom
#        short    right
    ctypedef struct FixedPoint:
      Fixed               x
      Fixed               y
    ctypedef struct FixedRect:
      Fixed               left
      Fixed               top
      Fixed               right
      Fixed               bottom
    ctypedef struct ATSTrapezoid:
      FixedPoint          upperLeft
      FixedPoint          upperRight
      FixedPoint          lowerRight
      FixedPoint          lowerLeft

    ctypedef unsigned long FourCharCode
    ctypedef FourCharCode ATSUFontVariationAxis
    ctypedef Fixed ATSUFontVariationValue
    ctypedef Fixed ATSUTextMeasurement


    #### Objects ###############################################################

    OSStatus ATSUCreateStyle(ATSUStyle * oStyle)
    OSStatus ATSUCreateAndCopyStyle(ATSUStyle iStyle, ATSUStyle* oStyle)
    OSStatus ATSUDisposeStyle(ATSUStyle iStyle)
    OSStatus ATSUSetStyleRefCon(ATSUStyle iStyle, unsigned int iRefCon)
    OSStatus ATSUGetStyleRefCon(ATSUStyle iStyle, unsigned int* oRefCon)
    OSStatus ATSUClearStyle(ATSUStyle iStyle)
    OSStatus ATSUSetAttributes(
        ATSUStyle               iStyle,
        unsigned int            iAttributeCount,
        ATSUAttributeTag *      iTag,
        ByteCount *             iValueSize,
        ATSUAttributeValuePtr * iValue)
    OSStatus ATSUGetAttribute(
        ATSUStyle               iStyle,
        ATSUAttributeTag        iTag,
        ByteCount               iExpectedValueSize,
        ATSUAttributeValuePtr   oValue,
        ByteCount *             oActualValueSize)

    OSStatus ATSUCreateTextLayout(ATSUTextLayout * oTextLayout)
    OSStatus ATSUCreateAndCopyTextLayout(
        ATSUTextLayout    iTextLayout,
        ATSUTextLayout *  oTextLayout)
    OSStatus ATSUGetLineControl(
        ATSUTextLayout          iTextLayout,
        UniCharArrayOffset      iLineStart,
        ATSUAttributeTag        iTag,
        ByteCount               iExpectedValueSize,
        ATSUAttributeValuePtr   oValue,
        ByteCount *             oActualValueSize)

    OSStatus ATSUCreateTextLayoutWithTextPtr(
        ConstUniCharArrayPtr   iText,
        UniCharArrayOffset     iTextOffset,
        UniCharCount           iTextLength,
        UniCharCount           iTextTotalLength,
        ItemCount              iNumberOfRuns,
        UniCharCount *         iRunLengths,
        ATSUStyle *            iStyles,
        ATSUTextLayout *       oTextLayout)
    OSStatus ATSUClearLayoutCache(
        ATSUTextLayout       iTextLayout,
        UniCharArrayOffset   iLineStart)
    OSStatus ATSUDisposeTextLayout(ATSUTextLayout iTextLayout)

    OSStatus ATSUSetTextPointerLocation(
      ATSUTextLayout         iTextLayout,
      ConstUniCharArrayPtr   iText,
      UniCharArrayOffset     iTextOffset,
      UniCharCount           iTextLength,
      UniCharCount           iTextTotalLength)
    OSStatus ATSUGetTextLocation(
      ATSUTextLayout        iTextLayout,
      void **               oText,
      int *             oTextIsStoredInHandle,
      UniCharArrayOffset *  oOffset,
      UniCharCount *        oTextLength,
      UniCharCount *        oTextTotalLength)

    OSStatus ATSUSetLayoutControls(
      ATSUTextLayout                iTextLayout,
      ItemCount                     iAttributeCount,
      ATSUAttributeTag *      iTag,
      ByteCount *             iValueSize,
      ATSUAttributeValuePtr * iValue)
    OSStatus ATSUGetLayoutControl(
      ATSUTextLayout          iTextLayout,
      ATSUAttributeTag        iTag,
      ByteCount               iExpectedValueSize,
      ATSUAttributeValuePtr   oValue,
      ByteCount *             oActualValueSize)

    OSStatus ATSUClearLayoutControls(
      ATSUTextLayout           iTextLayout,
      ItemCount                iTagCount,
      ATSUAttributeTag * iTag)

    OSStatus ATSUSetRunStyle(
      ATSUTextLayout       iTextLayout,
      ATSUStyle            iStyle,
      UniCharArrayOffset   iRunStart,
      UniCharCount         iRunLength)
    OSStatus ATSUGetRunStyle(
      ATSUTextLayout        iTextLayout,
      UniCharArrayOffset    iOffset,
      ATSUStyle *           oStyle,
      UniCharArrayOffset *  oRunStart,
      UniCharCount *        oRunLength)
    OSStatus ATSUGetContinuousAttributes(
      ATSUTextLayout       iTextLayout,
      UniCharArrayOffset   iOffset,
      UniCharCount         iLength,
      ATSUStyle            oStyle)

    OSStatus ATSUCreateFontFallbacks(ATSUFontFallbacks * oFontFallback)
    OSStatus ATSUDisposeFontFallbacks(ATSUFontFallbacks iFontFallbacks)
    OSStatus ATSUSetObjFontFallbacks(
      ATSUFontFallbacks        iFontFallbacks,
      ItemCount                iFontFallbacksCount,
      ATSUFontID *             iFonts,
      ATSUFontFallbackMethod   iFontFallbackMethod)
    OSStatus ATSUGetObjFontFallbacks(
      ATSUFontFallbacks         iFontFallbacks,
      ItemCount                 iMaxFontFallbacksCount,
      ATSUFontID *              oFonts,
      ATSUFontFallbackMethod *  oFontFallbackMethod,
      ItemCount *               oActualFallbacksCount)
    OSStatus ATSUSetTransientFontMatching(
      ATSUTextLayout   iTextLayout,
      Boolean          iTransientFontMatching)


    #### Fonts #################################################################

    OSStatus ATSUFontCount(ItemCount * oFontCount)
    OSStatus ATSUGetFontIDs(
      ATSUFontID * oFontIDs,
      ItemCount    iArraySize,
      ItemCount *  oFontCount)
    OSStatus ATSUCountFontNames(
      ATSUFontID   iFontID,
      ItemCount *  oFontNameCount)
    OSStatus ATSUGetIndFontName(
      ATSUFontID          iFontID,
      ItemCount           iFontNameIndex,
      ByteCount           iMaximumNameLength,
      Ptr                 oName,
      ByteCount *         oActualNameLength,
      FontNameCode *      oFontNameCode,
      FontPlatformCode *  oFontNamePlatform,
      FontScriptCode *    oFontNameScript,
      FontLanguageCode *  oFontNameLanguage)
    OSStatus ATSUFindFontName(
      ATSUFontID         iFontID,
      FontNameCode       iFontNameCode,
      FontPlatformCode   iFontNamePlatform,
      FontScriptCode     iFontNameScript,
      FontLanguageCode   iFontNameLanguage,
      ByteCount          iMaximumNameLength,
      Ptr                oName,
      ByteCount *        oActualNameLength,
      ItemCount *        oFontNameIndex)
    OSStatus ATSUFindFontFromName(
      void *       iName,
      ByteCount          iNameLength,
      FontNameCode       iFontNameCode,
      FontPlatformCode   iFontNamePlatform,
      FontScriptCode     iFontNameScript,
      FontLanguageCode   iFontNameLanguage,
      ATSUFontID *       oFontID)

    OSStatus ATSUCountFontInstances(
      ATSUFontID   iFontID,
      ItemCount *  oInstances)
    OSStatus ATSUGetFontInstance(
      ATSUFontID               iFontID,
      ItemCount                iFontInstanceIndex,
      ItemCount                iMaximumVariations,
      ATSUFontVariationAxis *  oAxes,
      ATSUFontVariationValue * oValues,
      ItemCount *              oActualVariationCount)
    OSStatus ATSUGetFontInstanceNameCode(
      ATSUFontID      iFontID,
      ItemCount       iInstanceIndex,
      FontNameCode *  oNameCode)


    #### Drawing ###############################################################

    OSStatus ATSUDrawText(
      ATSUTextLayout        iTextLayout,
      UniCharArrayOffset    iLineOffset,
      UniCharCount          iLineLength,
      ATSUTextMeasurement   iLocationX,
      ATSUTextMeasurement   iLocationY)
    OSStatus ATSUGetUnjustifiedBounds(
      ATSUTextLayout         iTextLayout,
      UniCharArrayOffset     iLineStart,
      UniCharCount           iLineLength,
      ATSUTextMeasurement *  oTextBefore,
      ATSUTextMeasurement *  oTextAfter,
      ATSUTextMeasurement *  oAscent,
      ATSUTextMeasurement *  oDescent)
    OSStatus ATSUMeasureTextImage(
      ATSUTextLayout        iTextLayout,
      UniCharArrayOffset    iLineOffset,
      UniCharCount          iLineLength,
      ATSUTextMeasurement   iLocationX,
      ATSUTextMeasurement   iLocationY,
      QDRect *                oTextImageRect)
    OSStatus ATSUGetGlyphBounds(
      ATSUTextLayout        iTextLayout,
      ATSUTextMeasurement   iTextBasePointX,
      ATSUTextMeasurement   iTextBasePointY,
      UniCharArrayOffset    iBoundsCharStart,
      UniCharCount          iBoundsCharLength,
      UInt16                iTypeOfBounds,
      ItemCount             iMaxNumberOfBounds,
      ATSTrapezoid *        oGlyphBounds,
      ItemCount *           oActualNumberOfBounds)
    OSStatus ATSUBatchBreakLines(
      ATSUTextLayout        iTextLayout,
      UniCharArrayOffset    iRangeStart,
      UniCharCount          iRangeLength,
      ATSUTextMeasurement   iLineWidth,
      ItemCount *           oBreakCount)
    OSStatus ATSUBreakLine(
      ATSUTextLayout        iTextLayout,
      UniCharArrayOffset    iLineStart,
      ATSUTextMeasurement   iLineWidth,
      Boolean               iUseAsSoftLineBreak,
      UniCharArrayOffset *  oLineBreak)
    OSStatus ATSUSetSoftLineBreak(
      ATSUTextLayout       iTextLayout,
      UniCharArrayOffset   iLineBreak)
    OSStatus ATSUGetSoftLineBreaks(
      ATSUTextLayout       iTextLayout,
      UniCharArrayOffset   iRangeStart,
      UniCharCount         iRangeLength,
      ItemCount            iMaximumBreaks,
      UniCharArrayOffset * oBreaks,
      ItemCount *          oBreakCount)
    OSStatus ATSUClearSoftLineBreaks(
      ATSUTextLayout       iTextLayout,
      UniCharArrayOffset   iRangeStart,
      UniCharCount         iRangeLength)


#### EOF #######################################################################
