#ifndef KIVA_AFFINE_MATRIX_H
#define KIVA_AFFINE_MATRIX_H

#include "agg_trans_affine.h"

namespace kiva
{
    bool is_identity(agg24::trans_affine& mat, double epsilon=1e-3);
    bool only_translation(agg24::trans_affine& mat, double epsilon=1e-3);
    bool only_scale_and_translation(agg24::trans_affine& mat, double epsilon=1e-3);
    void get_translation(agg24::trans_affine& m, double* tx, double* ty);
    void get_scale(agg24::trans_affine& m, double* dx, double* dy);
}

#endif
