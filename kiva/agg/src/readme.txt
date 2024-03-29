This directory contains the C++ source files and SWIG wrappers for Kiva's
Agg backend.

Support files
------------------------------
readme.txt          this file
todo.txt            Eric's old todo file

SWIG wrappers (*.i)
------------------------------
affine_matrix.i     agg_trans_affine.h classes
agg_std_string.i    typemaps for member access to font_type.name
agg_typemaps.i      typemaps for various agg classes (rect, color, etc.)
compiled_path.i     wrapper for kiva_compiled_path.h
constants.i         common enumerations and constants used by Agg and Kiva
font_type.i         wrapper for kiva_font_type.h
graphic_context.i   the main wrapper defining the Agg graphics context
numeric.i           typemaps and wrappers for Numpy array used in kiva
rect.i              wrapper for kiva_rect.h
rgba.i              RGBA color class and utility functions
rgba_array.i        maps Numeric 3- and 4-tuples into RGBA color instances
sequence_to_array.i maps Python tuples into double[]

swig_questions.txt  questions and problems we are currently having with our
                    use of SWIG

C/C++ files
-------------------------------
agg_examples.cpp    C++ source code for demonstrating use of various agg features
kiva_affine_helpers.h/.cpp
kiva_affine_matrix.h
kiva_basics.h
kiva_compiled_path.h/.cpp
kiva_constants.h
kiva_dash_type.h
kiva_exceptions.h
kiva_font_type.h
kiva_graphics_context_base.h/.cpp  non-templatized base class for graphics
                                   contexts (which are templatized on pixel
                                   format)
kiva_graphics_context.h            template graphics_context class and typedef
                                   specializations for various pixel formats.
kiva_image_filters.h    A helper class that associates the right types of
                        image filters for various pixel formats
kiva_pix_format.h       defines agg_pix_to_kiva()
kiva_rect.h/.cpp        Kiva rectangle class (with converters to/from double*,
                        Agg rects, etc.)

Directories
-------------------------------
gtk1     support files for grabbing a graphics context in GTK
win32       "      "    "     "     "     "       "     " win32
x11         "      "    "     "     "     "       "     " xwindows
osx         "      "    "     "     "     "       "     " OSX



