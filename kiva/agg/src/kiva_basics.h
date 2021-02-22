// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef KIVA_BASICS_H
#define KIVA_BASICS_H

#include "kiva_constants.h"

namespace kiva 
{

#ifdef _MSC_VER
    #if _MSC_VER => 1300
        typedef signed __int64      INT64, *PINT64;
    #else
        #ifndef INT64
            #define INT64 __int64
        #endif
    #endif
#endif
#ifdef __GNUC__
    typedef long long INT64;
#endif

#ifdef max
    #undef max
#endif
#ifdef min
    #undef min
#endif


    inline double max(double a, double b)
    {
        if (a>b) return a;
        else return b;
    }

    inline double min(double a, double b)
    {
        if (a<b) return a;
        else return b;
    }

    inline unsigned int quadrant(double x, double y)
    {
        if (x>=0)
        {
            if (y>=0) return 1;
            else return 4;
        }
        else
        {
            if (y>=0) return 2;
            else return 3;
        }
    }


    // Determines whether or not two floating point numbers are
    // essentially equal.  This uses an aspect of the IEEE floating-
    // point spec described in 
    // http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
    // The code for both functions are from this same page.
    //
    // The basic idea is to cast the floats to ints and look at then
    // number of floating point numbers between them.  This take advantage
    // of the fact that the IEEE representation puts the mantissa on the right,
    // meaning that incrementing an int representation of a float yields the
    // next representable float.  The two's complement stuff is to ensure
    // that comparisons across the 0 boundary work.

    // For floating point, a maxUlps (ULP=units in last place) is roughly
    // equivalent to a precision of 1/8,000,000 to 1/16,000,000
    inline bool almost_equal(float A, float B, int maxUlps = 100)
    {
        if (A==B) return true;
        // Make sure maxUlps is non-negative and small enough that the
        // default NAN won't compare as equal to anything.
        //assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);

        int aInt = *(int*)&A;
        // Make aInt lexicographically ordered as a twos-complement int
        if (aInt < 0)
            aInt = 0x80000000 - aInt;

        // Make bInt lexicographically ordered as a twos-complement int
        int bInt = *(int*)&B;
        if (bInt < 0)
            bInt = 0x80000000 - bInt;
        int intDiff = aInt - bInt;
        if (intDiff < 0)
            intDiff = -intDiff;
        if (intDiff <= maxUlps)
            return true;
        return false;
    }

    // For double, a maxUlps (ULP=units in last place) is roughly
    // equivalent to a precision of 1/4e15 to 1/8e15.
    inline bool almost_equal(double A, double B, int maxUlps = 10000)
    {
        // Make sure maxUlps is non-negative and small enough that the
        // default NAN won't compare as equal to anything.
        //assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);

        if (A==B) return true;

        INT64 aInt = *(INT64*)&A;
        // Make aInt lexicographically ordered as a twos-complement int
        if (aInt < 0)
            aInt = 0x80000000 - aInt;

        // Make bInt lexicographically ordered as a twos-complement int
        INT64 bInt = *(INT64*)&B;
        if (bInt < 0)
            bInt = 0x80000000 - bInt;
        INT64 intDiff = aInt - bInt;
        if (intDiff < 0)
            intDiff = -intDiff;
        if (intDiff <= maxUlps)
            return true;
        return false;
    }

}

#endif /* KIVA_BASICS_H */
