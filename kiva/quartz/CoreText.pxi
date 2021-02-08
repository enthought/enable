# (C) Copyright 2004-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

include "CoreFoundation.pxi"
include "CoreGraphics.pxi"

cdef extern from "ApplicationServices/ApplicationServices.h":

    ctypedef CFTypeRef CTFontRef
    ctypedef CFTypeRef CTFontCollectionRef
    ctypedef CFTypeRef CTFontDescriptorRef
    ctypedef CFTypeRef CTLineRef

    # Constants that are used throughout CoreText
    cdef CFStringRef kCTFontAttributeName
    cdef CFStringRef kCTKernAttributeName
    cdef CFStringRef kCTLigatureAttributeName
    cdef CFStringRef kCTForegroundColorAttributeName
    cdef CFStringRef kCTParagraphStyleAttributeName
    cdef CFStringRef kCTUnderlineStyleAttributeName
    cdef CFStringRef kCTVerticalFormsAttributeName
    cdef CFStringRef kCTGlyphInfoAttributeName
    
    ctypedef enum CTUnderlineStyle:
        kCTUnderlineStyleNone   = 0x00
        kCTUnderlineStyleSingle = 0x01
        kCTUnderlineStyleThick  = 0x02
        kCTUnderlineStyleDouble = 0x09

    ctypedef enum CTUnderlineStyleModifiers:
        kCTUnderlinePatternSolid        = 0x0000
        kCTUnderlinePatternDot          = 0x0100
        kCTUnderlinePatternDash         = 0x0200
        kCTUnderlinePatternDashDot      = 0x0300
        kCTUnderlinePatternDashDotDot   = 0x0400

    cdef CFStringRef kCTFontSymbolicTrait
    cdef CFStringRef kCTFontWeightTrait
    cdef CFStringRef kCTFontWidthTrait
    cdef CFStringRef kCTFontSlantTrait

    ctypedef enum:
        kCTFontClassMaskShift = 28

    ctypedef enum CTFontSymbolicTraits:
        kCTFontItalicTrait      = (1 << 0)
        kCTFontBoldTrait        = (1 << 1)
        kCTFontExpandedTrait    = (1 << 5)
        kCTFontCondensedTrait   = (1 << 6)
        kCTFontMonoSpaceTrait   = (1 << 10)
        kCTFontVerticalTrait    = (1 << 11)
        kCTFontUIOptimizedTrait = (1 << 12)
        kCTFontClassMaskTrait   = (15 << kCTFontClassMaskShift)

    ctypedef enum CTFontStylisticClass:
        kCTFontUnknownClass             = (0 << kCTFontClassMaskShift)
        kCTFontOldStyleSerifsClass      = (1 << kCTFontClassMaskShift)
        kCTFontTransitionalSerifsClass  = (2 << kCTFontClassMaskShift)
        kCTFontModernSerifsClass        = (3 << kCTFontClassMaskShift)
        kCTFontClarendonSerifsClass     = (4 << kCTFontClassMaskShift)
        kCTFontSlabSerifsClass          = (5 << kCTFontClassMaskShift)
        kCTFontFreeformSerifsClass      = (7 << kCTFontClassMaskShift)
        kCTFontSansSerifClass           = (8 << kCTFontClassMaskShift)
        kCTFontOrnamentalsClass         = (9 << kCTFontClassMaskShift)
        kCTFontScriptsClass             = (10 << kCTFontClassMaskShift)
        kCTFontSymbolicClass            = (12 << kCTFontClassMaskShift)

    cdef CFStringRef kCTFontNameAttribute
    cdef CFStringRef kCTFontDisplayNameAttribute
    cdef CFStringRef kCTFontFamilyNameAttribute
    cdef CFStringRef kCTFontStyleNameAttribute
    cdef CFStringRef kCTFontTraitsAttribute
    cdef CFStringRef kCTFontVariationAttribute
    cdef CFStringRef kCTFontSizeAttribute
    cdef CFStringRef kCTFontMatrixAttribute
    cdef CFStringRef kCTFontCascadeListAttribute
    cdef CFStringRef kCTFontCharacterSetAttribute
    cdef CFStringRef kCTFontLanguagesAttribute
    cdef CFStringRef kCTFontBaselineAdjustAttribute
    cdef CFStringRef kCTFontMacintoshEncodingsAttribute
    cdef CFStringRef kCTFontFeaturesAttribute
    cdef CFStringRef kCTFontFeatureSettingsAttribute
    cdef CFStringRef kCTFontFixedAdvanceAttribute
    cdef CFStringRef kCTFontOrientationAttribute
    
    ctypedef enum CTFontOrientation:
        kCTFontDefaultOrientation       = 0
        kCTFontHorizontalOrientation    = 1
        kCTFontVerticalOrientation      = 2

    cdef CFStringRef kCTFontCopyrightNameKey
    cdef CFStringRef kCTFontFamilyNameKey
    cdef CFStringRef kCTFontSubFamilyNameKey
    cdef CFStringRef kCTFontStyleNameKey
    cdef CFStringRef kCTFontUniqueNameKey
    cdef CFStringRef kCTFontFullNameKey
    cdef CFStringRef kCTFontVersionNameKey
    cdef CFStringRef kCTFontPostScriptNameKey
    cdef CFStringRef kCTFontTrademarkNameKey
    cdef CFStringRef kCTFontManufacturerNameKey
    cdef CFStringRef kCTFontDesignerNameKey
    cdef CFStringRef kCTFontDescriptionNameKey
    cdef CFStringRef kCTFontVendorURLNameKey
    cdef CFStringRef kCTFontDesignerURLNameKey
    cdef CFStringRef kCTFontLicenseNameKey
    cdef CFStringRef kCTFontLicenseURLNameKey
    cdef CFStringRef kCTFontSampleTextNameKey
    cdef CFStringRef kCTFontPostScriptCIDNameKey

    ctypedef enum CTFontUIFontType:
        kCTFontNoFontType                           = -1
        kCTFontUserFontType                         =  0
        kCTFontUserFixedPitchFontType               =  1
        kCTFontSystemFontType                       =  2
        kCTFontEmphasizedSystemFontType             =  3
        kCTFontSmallSystemFontType                  =  4
        kCTFontSmallEmphasizedSystemFontType        =  5
        kCTFontMiniSystemFontType                   =  6
        kCTFontMiniEmphasizedSystemFontType         =  7
        kCTFontViewsFontType                        =  8
        kCTFontApplicationFontType                  =  9
        kCTFontLabelFontType                        = 10
        kCTFontMenuTitleFontType                    = 11
        kCTFontMenuItemFontType                     = 12
        kCTFontMenuItemMarkFontType                 = 13
        kCTFontMenuItemCmdKeyFontType               = 14
        kCTFontWindowTitleFontType                  = 15
        kCTFontPushButtonFontType                   = 16
        kCTFontUtilityWindowTitleFontType           = 17
        kCTFontAlertHeaderFontType                  = 18
        kCTFontSystemDetailFontType                 = 19
        kCTFontEmphasizedSystemDetailFontType       = 20
        kCTFontToolbarFontType                      = 21
        kCTFontSmallToolbarFontType                 = 22
        kCTFontMessageFontType                      = 23
        kCTFontPaletteFontType                      = 24
        kCTFontToolTipFontType                      = 25
        kCTFontControlContentFontType               = 26

    cdef CFStringRef kCTFontVariationAxisIdentifierKey
    cdef CFStringRef kCTFontVariationAxisMinimumValueKey
    cdef CFStringRef kCTFontVariationAxisMaximumValueKey
    cdef CFStringRef kCTFontVariationAxisDefaultValueKey
    cdef CFStringRef kCTFontVariationAxisNameKey

    cdef CFStringRef kCTFontFeatureTypeIdentifierKey
    cdef CFStringRef kCTFontFeatureTypeNameKey
    cdef CFStringRef kCTFontFeatureTypeExclusiveKey
    cdef CFStringRef kCTFontFeatureTypeSelectorsKey
    cdef CFStringRef kCTFontFeatureSelectorIdentifierKey
    cdef CFStringRef kCTFontFeatureSelectorNameKey
    cdef CFStringRef kCTFontFeatureSelectorDefaultKey
    cdef CFStringRef kCTFontFeatureSelectorSettingKey

    #ctypedef enum CTFontTableTag:
    #    kCTFontTableBASE
    #    kCTFontTableCFF
    #    kCTFontTableDSIG
    #    kCTFontTableEBDT
    #    kCTFontTableEBLC
    #    kCTFontTableEBSC
    #    kCTFontTableGDEF
    #    kCTFontTableGPOS
    #    kCTFontTableGSUB
    #    kCTFontTableJSTF
    #    kCTFontTableLTSH
    #    kCTFontTableOS2
    #    kCTFontTablePCLT
    #    kCTFontTableVDMX
    #    kCTFontTableVORG
    #    kCTFontTableZapf
    #    kCTFontTableAcnt
    #    kCTFontTableAvar
    #    kCTFontTableBdat
    #    kCTFontTableBhed
    #    kCTFontTableBloc
    #    kCTFontTableBsln
    #    kCTFontTableCmap
    #    kCTFontTableCvar
    #    kCTFontTableCvt
    #    kCTFontTableFdsc
    #    kCTFontTableFeat
    #    kCTFontTableFmtx
    #    kCTFontTableFpgm
    #    kCTFontTableFvar
    #    kCTFontTableGasp
    #    kCTFontTableGlyf
    #    kCTFontTableGvar
    #    kCTFontTableHdmx
    #    kCTFontTableHead
    #    kCTFontTableHhea
    #    kCTFontTableHmtx
    #    kCTFontTableHsty
    #    kCTFontTableJust
    #    kCTFontTableKern
    #    kCTFontTableLcar
    #    kCTFontTableLoca
    #    kCTFontTableMaxp
    #    kCTFontTableMort
    #    kCTFontTableMorx
    #    kCTFontTableName
    #    kCTFontTableOpbd
    #    kCTFontTablePost
    #    kCTFontTablePrep
    #    kCTFontTableProp
    #    kCTFontTableTrak
    #    kCTFontTableVhea
    #    kCTFontTableVmtx
    #
    #ctypedef enum CTFontTableOptions:
    #    kCTFontTableOptionNoOptions = 0
    #    kCTFontTableOptionExcludeSynthetic = (1 << 0)
    #

    cdef CFStringRef kCTFontCollectionRemoveDuplicatesOption

    ctypedef enum CTLineTruncationType:
        kCTLineTruncationStart  = 0
        kCTLineTruncationEnd    = 1
        kCTLineTruncationMiddle = 2

    # Fonts
    CTFontRef CTFontCreateWithName(CFStringRef name, CGFloat size,
        void *matrix)
    CTFontRef CTFontCreateWithFontDescriptor(CTFontDescriptorRef descriptor,
        CGFloat size, void *matrix)
    
    #CTFontRef CTFontCreateUIFontForLanguage(CTFontUIFontType uiType,
    #    CGFloat size, CFStringRef language)
    #CTFontRef CTFontCreateCopyWithAttributes(CTFontRef font, CGFloat size,
    #    void *matrix, CTFontDescriptorRef attributes)
    CTFontRef CTFontCreateCopyWithSymbolicTraits(CTFontRef font, CGFloat size,
        void *matrix, CTFontSymbolicTraits symTraitValue, 
        CTFontSymbolicTraits symTraitMask)
    #CTFontRef CTFontCreateCopyWithFamily(CTFontRef font, CGFloat size,
    #    void *matrix, CFStringRef family)
    #CTFontRef CTFontCreateForString(CTFontRef currentFont, CFStringRef string,
    #    CFRange range)
    CTFontDescriptorRef CTFontCopyFontDescriptor(CTFontRef font)
    CFTypeRef CTFontCopyAttribute(CTFontRef font, CFStringRef attribute)
    CGFloat CTFontGetSize(CTFontRef font)
    void CTFontGetMatrix(CTFontRef font)
    CTFontSymbolicTraits CTFontGetSymbolicTraits(CTFontRef font)
    CFDictionaryRef CTFontCopyTraits(CTFontRef font)
    CFStringRef CTFontCopyPostScriptName(CTFontRef font)
    CFStringRef CTFontCopyFamilyName(CTFontRef font)
    CFStringRef CTFontCopyFullName(CTFontRef font)
    CFStringRef CTFontCopyDisplayName(CTFontRef font)
    CFStringRef CTFontCopyName(CTFontRef font, CFStringRef nameKey)
    CFStringRef CTFontCopyLocalizedName(CTFontRef font, CFStringRef nameKey,
        CFStringRef *language)
    #CFCharacterSetRef CTFontCopyCharacterSet(CTFontRef font)
    CFStringEncoding CTFontGetStringEncoding(CTFontRef font)
    CFArrayRef CTFontCopySupportedLanguages(CTFontRef font)
    Boolean CTFontGetGlyphsForCharacters(CTFontRef font,
        UniChar characters[], CGGlyph glyphs[], CFIndex count)
    CGFloat CTFontGetAscent(CTFontRef font)
    CGFloat CTFontGetDescent(CTFontRef font)
    CGFloat CTFontGetLeading(CTFontRef font)
    unsigned CTFontGetUnitsPerEm(CTFontRef font)
    CFIndex CTFontGetGlyphCount(CTFontRef font)
    CGRect CTFontGetBoundingBox(CTFontRef font)
    CGFloat CTFontGetUnderlinePosition(CTFontRef font)
    CGFloat CTFontGetUnderlineThickness(CTFontRef font)
    CGFloat CTFontGetSlantAngle(CTFontRef font)
    CGFloat CTFontGetCapHeight(CTFontRef font)
    CGFloat CTFontGetXHeight(CTFontRef font)
    CGGlyph CTFontGetGlyphWithName(CTFontRef font, CFStringRef glyphName)
    CGRect CTFontGetBoundingRectsForGlyphs(CTFontRef font,
        CTFontOrientation orientation, CGGlyph glyphs[],
        CGRect boundingRects[],CFIndex count)
    double CTFontGetAdvancesForGlyphs(CTFontRef font, CTFontOrientation orientation,
        CGGlyph glyphs[], CGSize advances[], CFIndex count)
    void CTFontGetVerticalTranslationsForGlyphs(CTFontRef font,
        CGGlyph glyphs[], CGSize translations[], CFIndex count)
    CGPathRef CTFontCreatePathForGlyph(CTFontRef font, CGGlyph glyph,
        void * transform)
    
    CFArrayRef CTFontCopyVariationAxes(CTFontRef font)
    CFDictionaryRef CTFontCopyVariation(CTFontRef font)
    
    CFArrayRef CTFontCopyFeatures(CTFontRef font)
    CFArrayRef CTFontCopyFeatureSettings(CTFontRef font)
    CGFontRef CTFontCopyGraphicsFont(CTFontRef font, CTFontDescriptorRef *attributes)
    CTFontRef CTFontCreateWithGraphicsFont(CGFontRef graphicsFont, CGFloat size,
        void *matrix, CTFontDescriptorRef attributes)
    #ATSFontRef CTFontGetPlatformFont(CTFontRef font, CTFontDescriptorRef *attributes)
    #CTFontRef CTFontCreateWithPlatformFont(ATSFontRef platformFont, CGFloat size,
    #    void *matrix, CTFontDescriptorRef attributes)
    #CTFontRef CTFontCreateWithQuickdrawInstance(ConstStr255Param name,
    #    int16_t identifier, uint8_t style, CGFloat size)
    
    #CFArrayRef CTFontCopyAvailableTables(CTFontRef font,
    #    CTFontTableOptions options)
    #CFDataRef CTFontCopyTable(CTFontRef font, CTFontTableTag table,
    #    CTFontTableOptions options)

    # Font Collections
    CTFontCollectionRef CTFontCollectionCreateFromAvailableFonts(CFDictionaryRef options)
    CTFontCollectionRef CTFontCollectionCreateWithFontDescriptors(CFArrayRef descriptors,
        CFDictionaryRef options)
    #CTFontCollectionRef CTFontCollectionCreateCopyWithFontDescriptors(
    #    CTFontCollectionRef original, CFArrayRef descriptors,
    #    CFDictionaryRef options)
    CFArrayRef CTFontCollectionCreateMatchingFontDescriptors(CTFontCollectionRef collection)
    #CFArrayRef CTFontCollectionCreateMatchingFontDescriptorsSortedWithCallback(
    #    CTFontCollectionRef collection,
    #    CTFontCollectionSortDescriptorsCallback sortCallback,
    #    void* refCon)

    # Font Descriptors
    CTFontDescriptorRef CTFontDescriptorCreateWithNameAndSize(CFStringRef name,
        CGFloat size)
    CTFontDescriptorRef CTFontDescriptorCreateWithAttributes(
        CFDictionaryRef attributes)
    CTFontDescriptorRef CTFontDescriptorCreateCopyWithAttributes(
        CTFontDescriptorRef original, CFDictionaryRef attributes)
    
    #CTFontDescriptorRef CTFontDescriptorCreateCopyWithVariation(
    #    CTFontDescriptorRef original, CFNumberRef variationIdentifier,
    #    CGFloat variationValue)
    #CTFontDescriptorRef CTFontDescriptorCreateCopyWithFeature(
    #    CTFontDescriptorRef original, CFNumberRef featureTypeIdentifier,
    #    CFNumberRef featureSelectorIdentifier)
    #CFArrayRef CTFontDescriptorCreateMatchingFontDescriptors(
    #    CTFontDescriptorRef descriptor, CFSetRef mandatoryAttributes)
    #CTFontDescriptorRef CTFontDescriptorCreateMatchingFontDescriptor(
    #    CTFontDescriptorRef descriptor, CFSetRef mandatoryAttributes)
    #CFDictionaryRef CTFontDescriptorCopyAttributes(CTFontDescriptorRef descriptor)
    CFTypeRef CTFontDescriptorCopyAttribute( CTFontDescriptorRef descriptor,
        CFStringRef attribute)
    #CFTypeRef CTFontDescriptorCopyLocalizedAttribute(
    #    CTFontDescriptorRef descriptor, CFStringRef attribute,
    #    CFStringRef *language)

    # Lines
    CTLineRef CTLineCreateWithAttributedString(CFAttributedStringRef string)
    CTLineRef CTLineCreateTruncatedLine(CTLineRef line, double width,
        CTLineTruncationType truncationType, CTLineRef truncationToken)
    CTLineRef CTLineCreateJustifiedLine(CTLineRef line, CGFloat justificationFactor,
        double justificationWidth)
    CFIndex CTLineGetGlyphCount(CTLineRef line)
    CFArrayRef CTLineGetGlyphRuns(CTLineRef line)
    CFRange CTLineGetStringRange(CTLineRef line)
    double CTLineGetPenOffsetForFlush(CTLineRef line, CGFloat flushFactor,
        double flushWidth)
    void CTLineDraw(CTLineRef line, CGContextRef context)
    CGRect CTLineGetImageBounds(CTLineRef line, CGContextRef context)
    double CTLineGetTypographicBounds(CTLineRef line, CGFloat* ascent,
        CGFloat* descent, CGFloat* leading)
    double CTLineGetTrailingWhitespaceWidth(CTLineRef line)
    CFIndex CTLineGetStringIndexForPosition(CTLineRef line, CGPoint position)
    CGFloat CTLineGetOffsetForStringIndex(CTLineRef line, CFIndex charIndex,
        CGFloat* secondaryOffset)
