#ifndef KIVA_GL_AFFINE_MATRIX_H
#define KIVA_GL_AFFINE_MATRIX_H

#include "agg_trans_affine.h"

namespace kiva_gl
{
    bool is_identity(kiva_gl_agg::trans_affine& mat, double epsilon=1e-3);
    bool only_translation(kiva_gl_agg::trans_affine& mat, double epsilon=1e-3);
    bool only_scale_and_translation(kiva_gl_agg::trans_affine& mat, double epsilon=1e-3);
    void get_translation(kiva_gl_agg::trans_affine& m, double* tx, double* ty);
    void get_scale(kiva_gl_agg::trans_affine& m, double* dx, double* dy);
}

#endif
