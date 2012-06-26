/* -*- c++ -*- */
/* File : example.i */
%module agg

#if (SWIG_VERSION > 0x010322)
%feature("compactdefaultargs");
#endif // (SWIG_VERSION > 0x010322)

%include "constants.i"
%include "rgba.i"
%include "font_type.i"
%include "affine_matrix.i"
%include "compiled_path.i"
//%include "image.i"
%include "graphics_context.i"
%include "hit_test.i"

