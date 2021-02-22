// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!

/////////////////////////////////////////////////////////////////////////////
//
// 1) Wraps constants and enumerated types commonly used in agg and kiva.
// 2) Provides typemaps to accpet integer inputs for enumerated types.
// 3) Provides python dictionaries to map back and forth between enumerated
//    types and more descriptive strings that can be used in python.
//
// A number of constants (and some functions and types) are defined in 
// agg_basics.h and kiva_constants.h.
//
// agg_renderer_markers.h is used for rendering simple shapes at multiple 
// data points.  It is useful for generating scatter plots in simple cases.
// This wrapper is used to pick up the enumerated types for markers such
// as marker_square, marker_circle, etc.  The only classes in the header are
// template definitions so they are all ignored by swig.
//
// 
/////////////////////////////////////////////////////////////////////////////

%{
#include "agg_basics.h"    
#include "kiva_constants.h"
#include "agg_renderer_markers.h"
%}

%include "agg_basics.h"
%include "kiva_constants.h"
%include "agg_renderer_markers.h"

%{
    inline unsigned path_cmd(unsigned c) { return c & agg24::path_cmd_mask; }
    inline unsigned path_flags(unsigned c) { return c & agg24::path_flags_mask; }
%}

%include "agg_typemaps.i"

%apply(kiva_enum_typemap) { agg24::path_flags_e };
%apply(kiva_enum_typemap) { agg24::marker_e };
%apply(kiva_enum_typemap) { kiva::draw_mode_e mode, kiva::text_draw_mode_e,
                            kiva::line_join_e, kiva::line_cap_e, kiva::blend_mode_e };
%apply(kiva_enum_typemap) { kiva::pix_format_e, kiva::interpolation_e };
%apply(kiva_enum_typemap) { kiva::blend_mode_e mode};

unsigned path_cmd(unsigned c);
unsigned path_flags(unsigned c);

%pythoncode %{

#----------------------------------------------------------------------------
#
# map strings values to the marker enumerated values and back with:
#   marker_string_map[string] = enum
#   marker_enum_map[enum] = string
#
#----------------------------------------------------------------------------

kiva_marker_to_agg = {}
kiva_marker_to_agg[1] = marker_square
kiva_marker_to_agg[2] = marker_diamond
kiva_marker_to_agg[3] = marker_circle
kiva_marker_to_agg[4] = marker_crossed_circle
kiva_marker_to_agg[5] = marker_x
kiva_marker_to_agg[6] = marker_triangle_up
kiva_marker_to_agg[7] = marker_triangle_down
kiva_marker_to_agg[8] = marker_cross    # "plus" sign; Agg calls this "cross"
kiva_marker_to_agg[9] = marker_dot
kiva_marker_to_agg[10] = marker_pixel


#----------------------------------------------------------------------------
#
# Map strings values to the pix_format enumerated values and back with:
#   pix_format_string_map[string] = enum
#   pix_format_enum_map[enum] = string
#
#----------------------------------------------------------------------------

pix_format_string_map = {}
pix_format_string_map["gray8"] = pix_format_gray8
pix_format_string_map["rgb555"] = pix_format_rgb555
pix_format_string_map["rgb565"] = pix_format_rgb565
pix_format_string_map["rgb24"] = pix_format_rgb24
pix_format_string_map["bgr24"] = pix_format_bgr24
pix_format_string_map["rgba32"] = pix_format_rgba32
pix_format_string_map["argb32"] = pix_format_argb32
pix_format_string_map["abgr32"] = pix_format_abgr32
pix_format_string_map["bgra32"] = pix_format_bgra32

pix_format_enum_map = {}
for key,value in pix_format_string_map.items():
    pix_format_enum_map[value] = key

#----------------------------------------------------------------------------
# Map a pix format string value to the number of bytes per pixel
#----------------------------------------------------------------------------

pix_format_bytes = {}
pix_format_bytes["gray8"] = 1
pix_format_bytes["rgb555"] = 2
pix_format_bytes["rgb565"] = 2
pix_format_bytes["rgb24"] = 3
pix_format_bytes["bgr24"] = 3
pix_format_bytes["rgba32"] = 4
pix_format_bytes["argb32"] = 4
pix_format_bytes["abgr32"] = 4
pix_format_bytes["bgra32"] = 4

pix_format_bits = {}
pix_format_bits["gray8"] = 8
pix_format_bits["rgb555"] = 15
pix_format_bits["rgb565"] = 16
pix_format_bits["rgb24"] = 24
pix_format_bits["bgr24"] = 24
pix_format_bits["rgba32"] = 32
pix_format_bits["argb32"] = 32
pix_format_bits["abgr32"] = 32
pix_format_bits["bgra32"] = 32

#----------------------------------------------------------------------------
#
# Map strings values to the interpolation enumerated values and back with:
#   interp_string_map[string] = enum
#   interp_enum_map[enum] = string
#
#----------------------------------------------------------------------------

interp_string_map = {}
interp_string_map["nearest"] = nearest
interp_string_map["bilinear"] = bilinear
interp_string_map["bicubic"] = bicubic
interp_string_map["spline16"] = spline16
interp_string_map["spline36"] = spline36
interp_string_map["sinc64"] = sinc64
interp_string_map["sinc144"] = sinc144
interp_string_map["sinc256"] = sinc256
interp_string_map["blackman64"] = blackman64
interp_string_map["blackman100"] = blackman100
interp_string_map["blackman256"] = blackman256

interp_enum_map = {}
for key,value in interp_string_map.items():
    interp_enum_map[value] = key

%}
