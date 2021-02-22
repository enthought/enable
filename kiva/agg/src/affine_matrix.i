/* -*- c -*- */
// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
/* AffineMatrix class wrapper

    1. C++ class 'trans_affine' is renamed to python '_AffineMatrix'

    2. All methods accept 'transform' and 'inverse_transform' are
       wrapped.

    3. __repr__ and __str__ methods are added to print out an
       _AffineMatrix object as: "AffineMatrix(a,b,c,d,tx,ty)"

    4. A subclass called 'AffineMatrix' is derived from '_AffineMatrix'
       using a %pythoncode directive.  This is so that __init__ can be
       overloadeded to convert a sequence into the appropriate argument
       convention for the _AffineMatrix constructor.

    5. Classes such as trans_affine_rotation were converted to factory
       functions so that they return an trans_affine class instead of
       having a new class type (such as RotationMatrix).

    Notes:
    !! 1.
    !! (4) is a hack to get around the fact that I couldn't
    !! figure out how to get the overloaded constructor for trans_affine
    !! to accept a Numeric array as input -- even if I added a function
    !! new_AffineMatrix(double ary[6]); and then put the
    !! trans_affine(double ary[6]) signature in the class interface.  It
    !! appears that SWIG is a little overzealous in its type checking
    !! in the constructor call, only allowing double* pointers through
    !! instead of allowing any sequence through.  This is the correct
    !! default behavior, but I couldn't figure out how to overload it with
    !! my own test.
    !!
    !! 2.
    !! The C++ operator *= is definitely broken -- probably not setting the
    !! thisown property correctly on returned pointers.  It is currently
    !! set to return void so that it can't cause any mischief, but it also
    !! breaks its functionality.
    !! FIX: I have just created this function in Python and call the
    !!      C++ multiply() method.
*/

%include "numeric.i"
%include "sequence_to_array.i"


%{
#ifdef NUMPY
#include "numpy/arrayobject.h"
#else
#include "Numeric/arrayobject.h"
#endif
#include "agg_trans_affine.h"

// These factories mimic the functionality of like-named classes in agg.
// Making them functions that return trans_affine types leads to a cleaner
// and easier to maintain Python interface.


agg24::trans_affine* trans_affine_rotation(double a)
{
    return new agg24::trans_affine(cos(a), sin(a), -sin(a), cos(a), 0.0, 0.0);
}

agg24::trans_affine* trans_affine_scaling(double sx, double sy)
{
    return new agg24::trans_affine(sx, 0.0, 0.0, sy, 0.0, 0.0);
}

agg24::trans_affine* trans_affine_translation(double tx, double ty)
{
    return new agg24::trans_affine(1.0, 0.0, 0.0, 1.0, tx, ty);
}

agg24::trans_affine* trans_affine_skewing(double sx, double sy)
{
    return new agg24::trans_affine(1.0, tan(sy), tan(sx), 1.0, 0.0, 0.0);
}

%}

%newobject trans_affine_rotation;
%rename(rotation_matrix) trans_affine_rotation(double);
agg24::trans_affine* trans_affine_rotation(double a);

%newobject trans_affine_scaling;
%rename(scaling_matrix) trans_affine_scaling(double, double);
agg24::trans_affine* trans_affine_scaling(double sx, double sy);

%newobject trans_affine_translation;
%rename(translation_matrix) trans_affine_translation(double, double);
agg24::trans_affine* trans_affine_translation(double tx, double ty);

%newobject trans_affine_skewing;
%rename(skewing_matrix) trans_affine_skewing(double, double);
agg24::trans_affine* trans_affine_skewing(double sx, double sy);


%include "agg_typemaps.i"
%apply (double* array6) {(double* out)};

// used by __getitem__
%typemap(check) (int affine_index)
{
    if ($1 < 0 || $1 > 5)
    {
         PyErr_Format(PyExc_IndexError,
                      "affine matrices are indexed 0 to 5. Received %d", $1);
         return NULL;
    }
}

%apply owned_pointer { agg24::trans_affine* };


namespace agg24
{
    %rename(_AffineMatrix) trans_affine;
    %rename(asarray) trans_affine::store_to(double*) const;

    class trans_affine
    {
    public:
        trans_affine();
        trans_affine(const trans_affine& m);
        trans_affine(double v0, double v1, double v2, double v3,
                      double v4, double v5);
        trans_affine operator ~ () const;
        // I added this to trans_affine -- it really isn't there.
        // trans_affine operator *(const trans_affine& m);

        // Returning trans_affine& causes problems, so these are all
        // changed to void.
        //const trans_affine& operator *= (const trans_affine& m);
        //const trans_affine& reset();
        // const trans_affine& multiply(const trans_affine& m);
        // const trans_affine& invert();
        // const trans_affine& flip_x();
        // const trans_affine& flip_y();
        //void operator *= (const trans_affine& m);
        void reset();
        void multiply(const trans_affine& m);
        void invert();
        void flip_x();
        void flip_y();

        double scale() const;
        double determinant() const;

        void store_to(double* out) const;
        //const trans_affine& load_from(double ary[6]);
        void load_from(double ary[6]);

        // !! omitted
        //void transform(double* x, double* y) const;
        //void inverse_transform(double* x, double* y) const;
    };
};

%pythoncode %{
def is_sequence(arg):
    try:
        len(arg)
        return 1
    except:
        return 0

# AffineMatrix sub-class to get around problems with adding
# a AffineMatrix constructor that accepts a Numeric array
# as input.
class AffineMatrix(_AffineMatrix):
    def __init__(self,*args):
        if len(args) == 1 and is_sequence(args[0]):
            args = tuple(args[0])
            if len(args) != 6:
                raise ValueError("array argument must be 1x6")
        _AffineMatrix.__init__(self,*args)

    def __imul__(self,other):
        """ inplace multiply

            We don't use the C++ version of this because it ends up
            deleting the object out from under itself.
        """
        self.multiply(other)
        return self
%}

%extend agg24::trans_affine
{
    char *__repr__()
    {
        // Write out elements of trans_affine in a,b,c,d,tx,ty order
        // !! We should work to make output formatting conform to
        // !! whatever it Numeric does (which needs to be cleaned up also).
        static char tmp[1024];
        double m[6];
        self->store_to(m);
        sprintf(tmp,"AffineMatrix(%g,%g,%g,%g,%g,%g)", m[0], m[1], m[2],
                                                        m[3], m[4], m[5]);
        return tmp;
    }
    double __getitem__(int affine_index)
    {
        double ary[6];
        self->store_to(ary);
        return ary[affine_index];
    }
    int __eq__(agg24::trans_affine& other)
    {
        double ary1[6], ary2[6];
        self->store_to(ary1);
        other.store_to(ary2);
        int eq = 1;
        for (int i = 0; i < 6; i++)
            eq &= (ary1[i] == ary2[i]);
        return eq;
    }
}

