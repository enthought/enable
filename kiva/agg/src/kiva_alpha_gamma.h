// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef KIVA_ALPHA_GAMMA_H
#define KIVA_ALPHA_GAMMA_H

#include "agg_gamma_functions.h"

namespace kiva
{
    struct alpha_gamma
    {
        alpha_gamma(double alpha, double gamma) :
            m_alpha(alpha), m_gamma(gamma) {}

        double operator() (double x) const
        {
            return m_alpha(m_gamma(x));
        }
        agg24::gamma_multiply m_alpha;
        agg24::gamma_power    m_gamma;
    };
}

#endif
