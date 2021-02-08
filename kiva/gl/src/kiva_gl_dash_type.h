// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef KIVA_GL_DASH_TYPE_H
#define KIVA_GL_DASH_TYPE_H

#include <vector>

namespace kiva_gl
{
    //-----------------------------------------------------------------------
    // line dash type
    //-----------------------------------------------------------------------

    class dash_type
    {
        public:
            double phase;
            std::vector<double> pattern;

            // constructor
            dash_type()
            : phase(0)
            , pattern(2, 0)
            {
            }

            // this forces even length of pattern
            dash_type(double _phase, double* _pattern, int n)
            : phase(_phase)
            , pattern(n % 2 ? n+1 : n)
            {
                for (int i = 0; i < n; ++i)
                {
                    pattern[i] = _pattern[i];
                }

                // for odd length patterns, use the first entry as the
                // last gap size. (this is arbitrary)
                if (n % 2)
                {
                    pattern[n] = _pattern[0];
                }
            }

            ~dash_type() {}

            bool is_solid()
            {
                return (pattern.size() == 2 && pattern[0] == 0.0);
            }

            // TODO-PZW: define a copy constructor
    };
}

#endif
