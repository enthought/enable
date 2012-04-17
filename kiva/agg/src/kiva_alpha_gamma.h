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
