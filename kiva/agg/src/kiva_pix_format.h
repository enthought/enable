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
// This set of templatized functions return a kiva::pix_format_e value based
// on the agg pixel format template parameter.  The function is called from 
// the graphics_context<T>.format() method.
//
/////////////////////////////////////////////////////////////////////////////

#ifndef KIVA_PIX_FORMAT_H
#define KIVA_PIX_FORMAT_H

#include "agg_pixfmt_gray.h"
#include "agg_pixfmt_rgb.h"
#include "agg_pixfmt_rgba.h"

#include "kiva_constants.h"

namespace kiva
{
    // Templatized conversion function to turn Agg pixel formats
    // into the right kiva pixfmt enumeration instance.
    // The default function returns "undefined", and the template
    // is specialized for each of the actual Agg pixel formats.
    template <class AggPixFmt> kiva::pix_format_e agg_pix_to_kiva(void *dummy=NULL)
    { 
        return kiva::pix_format_undefined;
    }

    // Specializations follow.
    // The dummy pointer argument is needed because MSVC++ 6.0 doesn't properly
    // support specialization of template functions.  When it generates code for
    // template functions, the mangled name of an instance of a function template
    // only includes the function parameter types (and not the template arguments).
    // The linker then discards the seemingly duplicate function definitions,
    // and all calls to agg_pix_to_kiva<T>() end up in an arbitrary one of the
    // following specializations (usually the first).  The obvious workaround is
    // to add optional function parameters corresponding to the template parameter.
    // 
    // Furthermore, when calling these functions, MSVC will complain about
    // ambiguous overloading, so you have to create a dummy pointer and pass it
    // in.  Normally, you would be able to do this:
    //
    //      template <agg_pix_fmt> void Foo()
    //      {
    //          do_something( agg_pix_to_kiva<agg_pix_fmt>() );
    //      }
    //
    // But with MSVC, you have to do this:
    //
    //      template <agg_pix_fmt> void Foo()
    //      {
    //          agg_pix_fmt *dummy = NULL;
    //          do_something( agg_pix_to_kiva(dummy) );
    //      }
    //
    //
    inline kiva::pix_format_e agg_pix_to_kiva(agg24::pixfmt_gray8 *msvc6_dummy = NULL)
    {
        return kiva::pix_format_gray8;
    }
    inline kiva::pix_format_e agg_pix_to_kiva(agg24::pixfmt_rgb24 *msvc6_dummy = NULL)
    {
        return kiva::pix_format_rgb24;
    }
    inline kiva::pix_format_e agg_pix_to_kiva(agg24::pixfmt_bgr24 *msvc6_dummy = NULL)
    {
        return kiva::pix_format_bgr24;
    }
    inline kiva::pix_format_e agg_pix_to_kiva(agg24::pixfmt_bgra32 *msvc6_dummy = NULL)
    {
        return kiva::pix_format_bgra32;
    }
    inline kiva::pix_format_e agg_pix_to_kiva(agg24::pixfmt_rgba32 *msvc6_dummy = NULL)
    {
        return kiva::pix_format_rgba32;
    }
    inline kiva::pix_format_e agg_pix_to_kiva(agg24::pixfmt_argb32 *msvc6_dummy = NULL)
    {
        return kiva::pix_format_argb32;
    }
    inline kiva::pix_format_e agg_pix_to_kiva(agg24::pixfmt_abgr32 *msvc6_dummy = NULL)
    {
        return kiva::pix_format_abgr32;
    }
}

#endif
