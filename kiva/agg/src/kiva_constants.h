// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef KIVA_CONSTANTS_H
#define KIVA_CONSTANTS_H

namespace kiva
{

    //-----------------------------------------------------------------------
    // Line Cap Constants
    //-----------------------------------------------------------------------

    enum line_cap_e
	{
		CAP_ROUND  = 0,
		CAP_BUTT   = 1,
		CAP_SQUARE = 2
    };

    //-----------------------------------------------------------------------
    // Line Join Constants
    //-----------------------------------------------------------------------

    enum line_join_e
	{
       JOIN_ROUND = 0,
       JOIN_BEVEL = 1,
       JOIN_MITER = 2
    };

    //-----------------------------------------------------------------------
    // Path Drawing Mode Constants
    //
    // Path drawing modes for path drawing methods.
    // The values are chosen so that bit flags can be checked in a later
    // C version.
    //-----------------------------------------------------------------------

    enum draw_mode_e
	{
        FILL            = 1,
        EOF_FILL        = 2,
        STROKE          = 4,
        FILL_STROKE     = 5,
        EOF_FILL_STROKE = 6
    };


    //-----------------------------------------------------------------------
    // Font Constants
    //
    // These are pretty much taken from wxPython.
    // !! Not sure if they are needed.
    //-----------------------------------------------------------------------

    enum text_style_e
	{
		NORMAL = 0,
		BOLD   = 1,
		ITALIC = 2
	};

    //-----------------------------------------------------------------------
    // Text Drawing Mode Constants
    //-----------------------------------------------------------------------

    enum text_draw_mode_e
	{
        TEXT_FILL               = 0,
        TEXT_STROKE             = 1,
        TEXT_FILL_STROKE        = 2,
        TEXT_INVISIBLE          = 3,
        TEXT_FILL_CLIP          = 4,
        TEXT_STROKE_CLIP        = 5,
        TEXT_FILL_STROKE_CLIP   = 6,
        TEXT_CLIP               = 7
    };


    //-----------------------------------------------------------------------
	// The following enums are Agg-specific, and might not be applicable
	// to other backends.
    //-----------------------------------------------------------------------

    enum interpolation_e
    {
        nearest = 0,
        bilinear = 1,
        bicubic = 2,
        spline16 = 3,
        spline36 = 4,
        sinc64 = 5,
        sinc144 = 6,
        sinc256 = 7,
        blackman64 = 8,
        blackman100 = 9,
        blackman256 = 10
    };

    enum pix_format_e
    {
        pix_format_undefined = 0,  // By default. No conversions are applied
        pix_format_gray8,          // Simple 256 level grayscale
        pix_format_rgb555,         // 15 bit rgb. Depends on the byte ordering!
        pix_format_rgb565,         // 16 bit rgb. Depends on the byte ordering!
        pix_format_rgb24,          // R-G-B, one byte per color component
        pix_format_bgr24,          // B-G-R, native win32 BMP format.
        pix_format_rgba32,         // R-G-B-A, one byte per color component
        pix_format_argb32,         // A-R-G-B, native MAC format
        pix_format_abgr32,         // A-B-G-R, one byte per color component
        pix_format_bgra32,         // B-G-R-A, native win32 BMP format

        end_of_pix_formats
    };

    enum blend_mode_e
    {
        blend_normal,        // pdf nrmal blending mode.
        blend_copy,          // overright destination with src ignoring any alpha setting.
        /*
        // these are copies from agg -- but not yet supported.
        blend_clear,         //----clear
        blend_src,           //----src
        blend_dst,           //----dst
        blend_src_over,      //----src_over
        blend_dst_over,      //----dst_over
        blend_src_in,        //----src_in
        blend_dst_in,        //----dst_in
        blend_src_out,       //----src_out
        blend_dst_out,       //----dst_out
        blend_src_atop,      //----src_atop
        blend_dst_atop,      //----dst_atop
        blend_xor,           //----xor
        blend_plus,          //----plus
        blend_minus,         //----minus
        blend_multiply,      //----multiply
        blend_screen,        //----screen
        blend_overlay,       //----overlay
        blend_darken,        //----darken
        blend_lighten,       //----lighten
        blend_color_dodge,   //----color_dodge
        blend_color_burn,    //----color_burn
        blend_hard_light,    //----hard_light
        blend_soft_light,    //----soft_light
        blend_difference,    //----difference
        blend_exclusion,     //----exclusion
        blend_contrast,      //----contrast
        */
        end_of_e
    };

    enum gradient_type_e
    {
        grad_none = 0,
        grad_linear,
        grad_radial
    };

    enum gradient_spread_e
    {
        pad = 0,
        reflect,
        repeat
    };

    enum gradient_units_e
    {
    	user_space = 0,
    	object_bounding_box
    };

}
#endif
