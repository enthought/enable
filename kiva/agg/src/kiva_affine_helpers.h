// (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
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
